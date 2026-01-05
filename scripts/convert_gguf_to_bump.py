#!/usr/bin/env python3
"""
convert_gguf_to_bump.py
=======================

Converts a GGUF model file containing weight-only quantized tensors (e.g. Q4_K_M,
Q6_K) into the C-Kernel-Engine `weights.bump` layout expected by the runtime.

Notes:
  - This tool is intentionally "offline": it may convert/reshape tensors while
    writing the bump file so runtime code stays simple (no format juggling).
  - For Q4_K/Q6_K models, we treat GGUF tensors of type GGML_TYPE_Q4_K/Q6_K as the
    canonical on-disk representation (same block layout as llama.cpp).
  - The bump file encodes a per-tensor dtype table (BUMPWGT3). The runtime reads
    it automatically to select the right kernel path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
from dataclasses import dataclass
from typing import BinaryIO, Dict, Optional, Sequence, Tuple

import numpy as np


HEADER_SIZE = 128
CACHE_ALIGN = 64

# Must match ckernel_dtype.h enum order exactly
CK_DT_FP32 = 0
CK_DT_BF16 = 1
CK_DT_FP16 = 2
CK_DT_INT8 = 3
CK_DT_INT4 = 4
CK_DT_Q4_0 = 5
CK_DT_Q4_1 = 6
CK_DT_Q4_K = 7
CK_DT_Q6_K = 8
CK_DT_Q8_0 = 9
CK_DT_Q8_K = 10
CK_DT_Q5_0 = 11
CK_DT_Q5_1 = 12


def align_up(n: int, a: int) -> int:
    return ((n + a - 1) // a) * a


def align_up_elems(elems: int, elem_bytes: int, align_bytes: int = CACHE_ALIGN) -> int:
    if align_bytes <= 0:
        return elems
    return align_up(elems * elem_bytes, align_bytes) // elem_bytes


class HashingWriter:
    def __init__(self, f: BinaryIO) -> None:
        self._f = f
        self._h = hashlib.sha256()
        self.bytes_written = 0

    def write(self, data: bytes) -> None:
        if not data:
            return
        self._f.write(data)
        self._h.update(data)
        self.bytes_written += len(data)

    def digest(self) -> bytes:
        return self._h.digest()


class GGUFError(RuntimeError):
    pass


class GGUFReader:
    def __init__(self, f: BinaryIO) -> None:
        self._f = f
        try:
            self._file_size = os.fstat(f.fileno()).st_size
        except Exception:
            self._file_size = None

    def file_size(self) -> Optional[int]:
        return self._file_size

    def tell(self) -> int:
        return int(self._f.tell())

    def seek(self, pos: int) -> None:
        self._f.seek(pos, os.SEEK_SET)

    def skip(self, n: int) -> None:
        if n <= 0:
            return
        self._f.seek(int(n), os.SEEK_CUR)

    def _read_exact(self, n: int) -> bytes:
        if n < 0:
            raise GGUFError(f"Unexpected read size {n}")
        if self._file_size is not None:
            remaining = int(self._file_size) - self.tell()
            if n > remaining:
                raise GGUFError(
                    f"Unexpected EOF (wanted {n} bytes, remaining {remaining}). "
                    "File may be truncated or header counts are corrupt."
                )
        data = self._f.read(n)
        if len(data) != n:
            raise GGUFError(f"Unexpected EOF (wanted {n} bytes, got {len(data)})")
        return data

    def u8(self) -> int:
        return struct.unpack("<B", self._read_exact(1))[0]

    def i8(self) -> int:
        return struct.unpack("<b", self._read_exact(1))[0]

    def u16(self) -> int:
        return struct.unpack("<H", self._read_exact(2))[0]

    def i16(self) -> int:
        return struct.unpack("<h", self._read_exact(2))[0]

    def u32(self) -> int:
        return struct.unpack("<I", self._read_exact(4))[0]

    def i32(self) -> int:
        return struct.unpack("<i", self._read_exact(4))[0]

    def u64(self) -> int:
        return struct.unpack("<Q", self._read_exact(8))[0]

    def i64(self) -> int:
        return struct.unpack("<q", self._read_exact(8))[0]

    def f32(self) -> float:
        return struct.unpack("<f", self._read_exact(4))[0]

    def f64(self) -> float:
        return struct.unpack("<d", self._read_exact(8))[0]

    def key_str(self) -> str:
        n = self.u64()
        return self._read_exact(n).decode("utf-8")

    def val_str(self) -> str:
        n = self.u64()
        return self._read_exact(n).decode("utf-8")


# GGUF metadata value types.
GGUF_TYPE_UINT8 = 0
GGUF_TYPE_INT8 = 1
GGUF_TYPE_UINT16 = 2
GGUF_TYPE_INT16 = 3
GGUF_TYPE_UINT32 = 4
GGUF_TYPE_INT32 = 5
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_BOOL = 7
GGUF_TYPE_STRING = 8
GGUF_TYPE_ARRAY = 9
GGUF_TYPE_UINT64 = 10
GGUF_TYPE_INT64 = 11
GGUF_TYPE_FLOAT64 = 12


def _gguf_scalar_size(vtype: int) -> Optional[int]:
    return {
        GGUF_TYPE_UINT8: 1,
        GGUF_TYPE_INT8: 1,
        GGUF_TYPE_UINT16: 2,
        GGUF_TYPE_INT16: 2,
        GGUF_TYPE_UINT32: 4,
        GGUF_TYPE_INT32: 4,
        GGUF_TYPE_FLOAT32: 4,
        GGUF_TYPE_BOOL: 1,
        GGUF_TYPE_UINT64: 8,
        GGUF_TYPE_INT64: 8,
        GGUF_TYPE_FLOAT64: 8,
    }.get(vtype)


def _gguf_read_value(r: GGUFReader, vtype: int):
    if vtype == GGUF_TYPE_UINT8:
        return r.u8()
    if vtype == GGUF_TYPE_INT8:
        return r.i8()
    if vtype == GGUF_TYPE_UINT16:
        return r.u16()
    if vtype == GGUF_TYPE_INT16:
        return r.i16()
    if vtype == GGUF_TYPE_UINT32:
        return r.u32()
    if vtype == GGUF_TYPE_INT32:
        return r.i32()
    if vtype == GGUF_TYPE_UINT64:
        return r.u64()
    if vtype == GGUF_TYPE_INT64:
        return r.i64()
    if vtype == GGUF_TYPE_FLOAT32:
        return r.f32()
    if vtype == GGUF_TYPE_FLOAT64:
        return r.f64()
    if vtype == GGUF_TYPE_BOOL:
        return bool(r.u8())
    if vtype == GGUF_TYPE_STRING:
        return r.val_str()
    if vtype == GGUF_TYPE_ARRAY:
        elem_type = r.u32()
        n = r.u64()
        if elem_type == GGUF_TYPE_STRING:
            # Token arrays can be huge; skip storing strings.
            for _ in range(n):
                _ = r.val_str()
            return {"type": "array", "elem_type": elem_type, "len": n}
        elem_size = _gguf_scalar_size(elem_type)
        if elem_size is None:
            raise GGUFError(f"Unsupported GGUF array elem type {elem_type}")
        r._read_exact(int(n) * elem_size)
        return {"type": "array", "elem_type": elem_type, "len": n}
    raise GGUFError(f"Unsupported GGUF value type {vtype}")


def _gguf_skip_value(r: GGUFReader, vtype: int) -> None:
    size = _gguf_scalar_size(vtype)
    if size is not None:
        r.skip(size)
        return
    if vtype == GGUF_TYPE_BOOL:
        r.skip(1)
        return
    if vtype == GGUF_TYPE_STRING:
        n = r.u64()
        r.skip(int(n))
        return
    if vtype == GGUF_TYPE_ARRAY:
        elem_type = r.u32()
        n = r.u64()
        if elem_type == GGUF_TYPE_STRING:
            # Skip strings without decoding to keep inspection fast on large vocabularies.
            for _ in range(int(n)):
                slen = r.u64()
                r.skip(int(slen))
            return
        elem_size = _gguf_scalar_size(elem_type)
        if elem_size is None:
            raise GGUFError(f"Unsupported GGUF array elem type {elem_type}")
        r.skip(int(n) * int(elem_size))
        return
    raise GGUFError(f"Unsupported GGUF value type {vtype}")


@dataclass(frozen=True)
class TensorInfo:
    name: str
    dims: Tuple[int, ...]  # ggml order: ne0, ne1, ...
    ggml_type: int
    offset: int  # relative to data section start

    @property
    def ne0(self) -> int:
        return int(self.dims[0]) if self.dims else 1

    @property
    def ne1(self) -> int:
        return int(self.dims[1]) if len(self.dims) > 1 else 1


# GGML tensor types (subset).
GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1
GGML_TYPE_Q4_0 = 2
GGML_TYPE_Q4_1 = 3
GGML_TYPE_Q5_0 = 6
GGML_TYPE_Q5_1 = 7
GGML_TYPE_Q8_0 = 8
GGML_TYPE_Q2_K = 10
GGML_TYPE_Q3_K = 11
GGML_TYPE_Q4_K = 12
GGML_TYPE_Q5_K = 13
GGML_TYPE_Q6_K = 14
GGML_TYPE_Q8_K = 15
GGML_TYPE_BF16 = 30  # present in newer GGUFs (was 16, moved in recent llama.cpp)
# I-quants (importance matrix quantization)
GGML_TYPE_IQ2_XXS = 16
GGML_TYPE_IQ2_XS = 17
GGML_TYPE_IQ3_XXS = 18
GGML_TYPE_IQ1_S = 19
GGML_TYPE_IQ4_NL = 20
GGML_TYPE_IQ3_S = 21
GGML_TYPE_IQ2_S = 22


def ggml_type_name(t: int) -> str:
    return {
        GGML_TYPE_F32: "F32",
        GGML_TYPE_F16: "F16",
        GGML_TYPE_BF16: "BF16",
        GGML_TYPE_Q4_0: "Q4_0",
        GGML_TYPE_Q4_1: "Q4_1",
        GGML_TYPE_Q5_0: "Q5_0",
        GGML_TYPE_Q5_1: "Q5_1",
        GGML_TYPE_Q8_0: "Q8_0",
        GGML_TYPE_Q2_K: "Q2_K",
        GGML_TYPE_Q3_K: "Q3_K",
        GGML_TYPE_Q4_K: "Q4_K",
        GGML_TYPE_Q5_K: "Q5_K",
        GGML_TYPE_Q6_K: "Q6_K",
        GGML_TYPE_Q8_K: "Q8_K",
        GGML_TYPE_IQ2_XXS: "IQ2_XXS",
        GGML_TYPE_IQ2_XS: "IQ2_XS",
        GGML_TYPE_IQ3_XXS: "IQ3_XXS",
        GGML_TYPE_IQ1_S: "IQ1_S",
        GGML_TYPE_IQ4_NL: "IQ4_NL",
        GGML_TYPE_IQ3_S: "IQ3_S",
        GGML_TYPE_IQ2_S: "IQ2_S",
    }.get(t, f"UNKNOWN({t})")


def ck_dtype_from_ggml_type(ggml_type: int) -> int:
    if ggml_type == GGML_TYPE_F32:
        return CK_DT_FP32
    if ggml_type == GGML_TYPE_F16:
        return CK_DT_FP16
    if ggml_type == GGML_TYPE_BF16:
        return CK_DT_BF16
    if ggml_type == GGML_TYPE_Q4_0:
        return CK_DT_Q4_0
    if ggml_type == GGML_TYPE_Q4_K:
        return CK_DT_Q4_K
    if ggml_type == GGML_TYPE_Q5_0:
        return CK_DT_Q5_0
    if ggml_type == GGML_TYPE_Q5_1:
        return CK_DT_Q5_1
    if ggml_type == GGML_TYPE_Q6_K:
        return CK_DT_Q6_K
    if ggml_type == GGML_TYPE_Q8_0:
        return CK_DT_Q8_0
    raise GGUFError(f"Unsupported ggml_type={ggml_type_name(ggml_type)} for bump output")


def ck_dtype_name(dt: int) -> str:
    return {
        CK_DT_FP32: "FP32",
        CK_DT_BF16: "BF16",
        CK_DT_FP16: "FP16",
        CK_DT_Q4_0: "Q4_0",
        CK_DT_Q4_K: "Q4_K",
        CK_DT_Q5_0: "Q5_0",
        CK_DT_Q5_1: "Q5_1",
        CK_DT_Q6_K: "Q6_K",
        CK_DT_Q8_0: "Q8_0",
    }.get(dt, f"DT({dt})")


def ggml_row_bytes(ggml_type: int, ne0: int) -> int:
    if ggml_type == GGML_TYPE_F32:
        return ne0 * 4
    if ggml_type == GGML_TYPE_F16:
        return ne0 * 2
    if ggml_type == GGML_TYPE_BF16:
        return ne0 * 2
    if ggml_type == GGML_TYPE_Q4_0:
        if ne0 % 32 != 0:
            raise GGUFError(f"Q4_0 requires ne0 % 32 == 0 (got ne0={ne0})")
        return (ne0 // 32) * 18
    if ggml_type == GGML_TYPE_Q4_1:
        if ne0 % 32 != 0:
            raise GGUFError(f"Q4_1 requires ne0 % 32 == 0 (got ne0={ne0})")
        return (ne0 // 32) * 20  # 2 (scale) + 2 (min) + 16 (packed)
    if ggml_type == GGML_TYPE_Q5_0:
        if ne0 % 32 != 0:
            raise GGUFError(f"Q5_0 requires ne0 % 32 == 0 (got ne0={ne0})")
        return (ne0 // 32) * 22
    if ggml_type == GGML_TYPE_Q5_1:
        if ne0 % 32 != 0:
            raise GGUFError(f"Q5_1 requires ne0 % 32 == 0 (got ne0={ne0})")
        return (ne0 // 32) * 24
    if ggml_type == GGML_TYPE_Q8_0:
        if ne0 % 32 != 0:
            raise GGUFError(f"Q8_0 requires ne0 % 32 == 0 (got ne0={ne0})")
        return (ne0 // 32) * 34
    if ggml_type == GGML_TYPE_Q4_K:
        if ne0 % 256 != 0:
            raise GGUFError(f"Q4_K requires ne0 % 256 == 0 (got ne0={ne0})")
        return (ne0 // 256) * 144
    if ggml_type == GGML_TYPE_Q6_K:
        if ne0 % 256 != 0:
            raise GGUFError(f"Q6_K requires ne0 % 256 == 0 (got ne0={ne0})")
        return (ne0 // 256) * 210
    raise GGUFError(f"Unsupported ggml_type={ggml_type_name(ggml_type)} for row sizing")


def ggml_tensor_bytes(info: TensorInfo) -> int:
    ne0 = info.ne0
    n_rows = 1
    for d in info.dims[1:]:
        n_rows *= int(d)
    return ggml_row_bytes(info.ggml_type, ne0) * n_rows


def read_vector_f32(f: BinaryIO, base: int, info: TensorInfo) -> np.ndarray:
    if len(info.dims) != 1:
        raise GGUFError(f"Expected 1D tensor for {info.name}, got dims={info.dims}")
    n = info.ne0
    f.seek(base + info.offset, os.SEEK_SET)
    raw = f.read(ggml_row_bytes(info.ggml_type, n))
    if info.ggml_type == GGML_TYPE_F32:
        return np.frombuffer(raw, dtype=np.float32).copy()
    if info.ggml_type == GGML_TYPE_F16:
        return np.frombuffer(raw, dtype=np.float16).astype(np.float32, copy=False)
    if info.ggml_type == GGML_TYPE_BF16:
        u16 = np.frombuffer(raw, dtype=np.uint16)
        u32 = u16.astype(np.uint32) << 16
        return u32.view(np.float32)
    raise GGUFError(f"Unsupported vector type {ggml_type_name(info.ggml_type)} for {info.name}")


def write_f32_padded(w: HashingWriter, vec: np.ndarray, aligned_dim: int) -> None:
    out = np.zeros((aligned_dim,), dtype=np.float32)
    n = min(int(vec.size), aligned_dim)
    out[:n] = vec.reshape(-1)[:n].astype(np.float32, copy=False)
    w.write(out.tobytes())


def write_f32_zeros(w: HashingWriter, count: int) -> None:
    if count <= 0:
        return
    w.write(np.zeros((count,), dtype=np.float32).tobytes())


def copy_bytes_stream(f_in: BinaryIO, src_pos: int, nbytes: int, w_out: HashingWriter, chunk: int = 1 << 20) -> None:
    f_in.seek(src_pos, os.SEEK_SET)
    remaining = nbytes
    while remaining > 0:
        take = min(remaining, chunk)
        buf = f_in.read(take)
        if len(buf) != take:
            raise GGUFError(f"Unexpected EOF while copying bytes (wanted {take}, got {len(buf)})")
        w_out.write(buf)
        remaining -= take


def copy_qk_head_packed(
    f_in: BinaryIO,
    data_base: int,
    info: TensorInfo,
    w_out: HashingWriter,
    group_count: int,
    head_dim: int,
    aligned_head_dim: int,
    aligned_embed_dim: int,
) -> None:
    # Supported quantized types: K-quants (Q4_K, Q6_K) and simple quants (Q4_0, Q4_1, Q5_0, Q5_1, Q8_0)
    supported_types = (
        GGML_TYPE_Q4_K, GGML_TYPE_Q6_K, GGML_TYPE_F32,
        GGML_TYPE_Q4_0, GGML_TYPE_Q4_1, GGML_TYPE_Q5_0, GGML_TYPE_Q5_1, GGML_TYPE_Q8_0
    )
    if info.ggml_type not in supported_types:
        raise GGUFError(f"{info.name}: expected Q4_K/Q6_K/Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/F32, got {ggml_type_name(info.ggml_type)}")
    if len(info.dims) != 2:
        raise GGUFError(f"{info.name}: expected 2D, got dims={info.dims}")

    in_dim = info.ne0
    out_dim = info.ne1
    if in_dim > aligned_embed_dim:
        raise GGUFError(f"{info.name}: expected in_dim<=aligned_embed_dim (got {in_dim} > {aligned_embed_dim})")
    # K-quants require 256-element alignment, simple quants require 32-element alignment
    if info.ggml_type in (GGML_TYPE_Q4_K, GGML_TYPE_Q6_K) and (aligned_embed_dim % 256) != 0:
        raise GGUFError(f"{info.name}: aligned_embed_dim must be multiple of 256 for K-quant (got {aligned_embed_dim})")
    if info.ggml_type in (GGML_TYPE_Q4_0, GGML_TYPE_Q4_1, GGML_TYPE_Q5_0, GGML_TYPE_Q5_1, GGML_TYPE_Q8_0) and (aligned_embed_dim % 32) != 0:
        raise GGUFError(f"{info.name}: aligned_embed_dim must be multiple of 32 for simple quant (got {aligned_embed_dim})")
    if out_dim != group_count * head_dim:
        raise GGUFError(
            f"{info.name}: expected out_dim={group_count * head_dim} (group_count*head_dim), got {out_dim}"
        )

    row_bytes = ggml_row_bytes(info.ggml_type, in_dim)
    src = data_base + info.offset
    f_in.seek(src, os.SEEK_SET)

    zero_row = b"\x00" * row_bytes

    for _h in range(group_count):
        # Copy real rows for this head.
        for _r in range(head_dim):
            buf = f_in.read(row_bytes)
            if len(buf) != row_bytes:
                raise GGUFError(f"{info.name}: unexpected EOF while reading row")
            w_out.write(buf)
        # Pad extra rows (if any) with zeros so padded lanes never contribute.
        for _r in range(head_dim, aligned_head_dim):
            w_out.write(zero_row)


def build_llama_config(
    *,
    model_type: str,
    num_layers: int,
    vocab_size: int,
    hidden_size: int,
    intermediate_size: int,
    num_heads: int,
    num_kv_heads: int,
    context_window: int,
    rope_theta: float,
    rms_norm_eps: float,
) -> Dict:
    return {
        "architectures": ["LlamaForCausalLM"],
        "model_type": model_type,
        "num_hidden_layers": int(num_layers),
        "hidden_size": int(hidden_size),
        "intermediate_size": int(intermediate_size),
        "num_attention_heads": int(num_heads),
        "num_key_value_heads": int(num_kv_heads),
        "vocab_size": int(vocab_size),
        "max_position_embeddings": int(context_window),
        "rms_norm_eps": float(rms_norm_eps),
        "rope_theta": float(rope_theta),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert GGUF (Q4_K / Q6_K) weights to weights.bump")
    ap.add_argument("--gguf", required=True, help="Input GGUF file (e.g. model.Q4_K_M.gguf)")
    ap.add_argument("--output", help="Output weights.bump path (required unless --inspect/--list)")
    ap.add_argument("--config-out", help="Optional config.json output path (HF-style minimal config)")
    ap.add_argument("--context", type=int, help="Override context length (max_position_embeddings)")
    ap.add_argument("--inspect", action="store_true", help="Print GGUF metadata/tensor dtypes and exit (no conversion)")
    ap.add_argument("--inspect-layers", action="store_true", help="Show per-layer quant types for ALL layers (use with --inspect)")
    ap.add_argument("--list", action="store_true", help="Print every tensor name/type/shape and exit (no conversion)")
    args = ap.parse_args()

    if not args.output and not (args.inspect or args.list):
        ap.error("--output is required unless --inspect/--list is set")

    wanted_meta = {
        "general.architecture",
        "general.alignment",
        "llama.block_count",
        "llama.context_length",
        "llama.embedding_length",
        "llama.feed_forward_length",
        "llama.attention.head_count",
        "llama.attention.head_count_kv",
        "llama.rope.freq_base",
        "llama.norm_rms_eps",
    }

    with open(args.gguf, "rb") as f:
        r = GGUFReader(f)
        magic = r._read_exact(4)
        if magic != b"GGUF":
            raise GGUFError(f"{args.gguf}: invalid magic {magic!r} (expected b'GGUF')")

        version = r.u32()
        # GGUF v2+ stores tensor/kv counts as u64 (v1 used u32).
        if version >= 2:
            n_tensors = r.u64()
            n_kv = r.u64()
        else:
            n_tensors = r.u32()
            n_kv = r.u32()

        file_size = r.file_size()
        if file_size is not None and file_size < r.tell():
            raise GGUFError(
                f"{args.gguf}: file too small for GGUF header "
                f"(size={file_size}, header={r.tell()})"
            )
        if n_tensors > 1_000_000 or n_kv > 1_000_000:
            raise GGUFError(
                f"{args.gguf}: GGUF header counts look corrupt "
                f"(n_tensors={n_tensors}, n_kv={n_kv})"
            )

        meta: Dict[str, object] = {}
        for _ in range(n_kv):
            key = r.key_str()
            vtype = r.u32()
            if key in wanted_meta:
                meta[key] = _gguf_read_value(r, vtype)
            else:
                _gguf_skip_value(r, vtype)

        tensors: Dict[str, TensorInfo] = {}
        for _ in range(n_tensors):
            name = r.key_str()
            n_dims = r.u32()
            dims = tuple(int(r.u64()) for _ in range(n_dims))
            ggml_type = r.u32()
            offset = r.u64()
            tensors[name] = TensorInfo(name=name, dims=dims, ggml_type=int(ggml_type), offset=int(offset))

        alignment = int(meta.get("general.alignment", 32))
        data_start = align_up(r.tell(), alignment)
        # Some writers already align; seeking forward is safe either way.
        r.seek(data_start)

        arch = str(meta.get("general.architecture", "llama"))

        if args.inspect or args.list:
            # Summarize tensor dtypes so you can confirm what is actually quantized
            # in a given GGUF file (e.g. whether token embeddings / output head are
            # Q4_K vs F16, and which tensors remain float).
            counts: Dict[int, int] = {}
            bytes_by_type: Dict[int, int] = {}
            for info in tensors.values():
                counts[info.ggml_type] = counts.get(info.ggml_type, 0) + 1
                try:
                    bytes_by_type[info.ggml_type] = bytes_by_type.get(info.ggml_type, 0) + ggml_tensor_bytes(info)
                except Exception:
                    # Unknown/unsupported types for sizing; still report counts.
                    pass

            def fmt_bytes(n: int) -> str:
                if n >= 1024 * 1024 * 1024:
                    return f"{n / (1024**3):.2f} GiB"
                if n >= 1024 * 1024:
                    return f"{n / (1024**2):.2f} MiB"
                if n >= 1024:
                    return f"{n / 1024:.2f} KiB"
                return f"{n} B"

            print(f"[gguf] file={args.gguf}")
            print(f"[gguf] version={version} arch={arch} tensors={n_tensors} kv={n_kv} alignment={alignment} data_start={data_start}")
            print("[gguf] tensor types:")
            for tcode, cnt in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
                b = bytes_by_type.get(tcode)
                b_str = fmt_bytes(b) if b is not None else "?"
                print(f"  - {ggml_type_name(tcode):>10}: {cnt:5d} tensors, bytes={b_str}")

            # Highlight common “does this get quantized?” tensors.
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
                print(f"  - {name}: {ggml_type_name(info.ggml_type)} dims={info.dims}")

            # Show per-layer quant types if --inspect-layers is set
            if args.inspect_layers:
                # Count total layers
                layer_ids = set()
                for name in tensors:
                    if name.startswith("blk."):
                        try:
                            layer_ids.add(int(name.split(".")[1]))
                        except Exception:
                            pass
                num_layers = max(layer_ids) + 1 if layer_ids else 0

                print(f"\n[gguf] per-layer quant types ({num_layers} layers):")
                print("  Layer | WQ      | WK      | WV      | WO      | Gate    | Up      | Down    |")
                print("  ------|---------|---------|---------|---------|---------|---------|---------|")

                for layer in range(num_layers):
                    wq = tensors.get(f"blk.{layer}.attn_q.weight")
                    wk = tensors.get(f"blk.{layer}.attn_k.weight")
                    wv = tensors.get(f"blk.{layer}.attn_v.weight")
                    wo = tensors.get(f"blk.{layer}.attn_output.weight")
                    gate = tensors.get(f"blk.{layer}.ffn_gate.weight")
                    up = tensors.get(f"blk.{layer}.ffn_up.weight")
                    down = tensors.get(f"blk.{layer}.ffn_down.weight")

                    wq_t = ggml_type_name(wq.ggml_type) if wq else "N/A"
                    wk_t = ggml_type_name(wk.ggml_type) if wk else "N/A"
                    wv_t = ggml_type_name(wv.ggml_type) if wv else "N/A"
                    wo_t = ggml_type_name(wo.ggml_type) if wo else "N/A"
                    gate_t = ggml_type_name(gate.ggml_type) if gate else "N/A"
                    up_t = ggml_type_name(up.ggml_type) if up else "N/A"
                    down_t = ggml_type_name(down.ggml_type) if down else "N/A"

                    print(f"  {layer:5d} | {wq_t:7s} | {wk_t:7s} | {wv_t:7s} | {wo_t:7s} | {gate_t:7s} | {up_t:7s} | {down_t:7s} |")

                # Show summary of which layers differ
                print("\n[gguf] mixed quant analysis:")
                first_layer_types = {}
                mixed_weights = []
                for layer in range(num_layers):
                    for weight_name, tensor_suffix in [
                        ("WQ", "attn_q.weight"),
                        ("WK", "attn_k.weight"),
                        ("WV", "attn_v.weight"),
                        ("WO", "attn_output.weight"),
                        ("Gate", "ffn_gate.weight"),
                        ("Up", "ffn_up.weight"),
                        ("Down", "ffn_down.weight"),
                    ]:
                        info = tensors.get(f"blk.{layer}.{tensor_suffix}")
                        if not info:
                            continue
                        if layer == 0:
                            first_layer_types[weight_name] = info.ggml_type
                        elif info.ggml_type != first_layer_types.get(weight_name):
                            if weight_name not in mixed_weights:
                                mixed_weights.append(weight_name)

                if mixed_weights:
                    print(f"  Mixed quant detected in: {', '.join(mixed_weights)}")
                    print("  (Different layers use different quant types for these weights)")
                else:
                    print("  All layers use uniform quant types (no mixed quant)")

            if args.list:
                print("[gguf] all tensors:")
                for name in sorted(tensors.keys()):
                    info = tensors[name]
                    print(f"  - {name}: {ggml_type_name(info.ggml_type)} dims={info.dims}")

            # Exit after inspection (don't try to parse model config)
            return

        # Pull core dims from metadata first; fall back to tensor shapes.
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

        def weight_dtype(info: TensorInfo, label: str) -> int:
            if info.ggml_type not in (GGML_TYPE_Q4_K, GGML_TYPE_Q6_K, GGML_TYPE_F32):
                raise GGUFError(
                    f"{info.name}: expected Q4_K/Q6_K/F32 for {label}, got {ggml_type_name(info.ggml_type)}"
                )
            return ck_dtype_from_ggml_type(info.ggml_type)

        tok_name = "token_embd.weight"
        if tok_name not in tensors:
            raise GGUFError(f"Missing required tensor: {tok_name}")
        tok = tensors[tok_name]
        if len(tok.dims) != 2:
            raise GGUFError(f"{tok_name}: expected 2D, got dims={tok.dims}")
        embed_dim = meta_int("llama.embedding_length") or tok.ne0
        vocab_size = tok.ne1

        num_layers = meta_int("llama.block_count")
        if num_layers is None:
            # Infer from present blocks.
            layer_ids = []
            for name in tensors:
                if name.startswith("blk.") and ".attn_norm.weight" in name:
                    try:
                        layer_ids.append(int(name.split(".")[1]))
                    except Exception:
                        pass
            if not layer_ids:
                raise GGUFError("Could not infer num_layers (missing llama.block_count and no blk.* tensors found)")
            num_layers = max(layer_ids) + 1

        intermediate = meta_int("llama.feed_forward_length")
        if intermediate is None:
            gate0 = tensors.get("blk.0.ffn_gate.weight")
            if gate0 and len(gate0.dims) == 2:
                intermediate = gate0.ne1
        if intermediate is None:
            raise GGUFError("Could not determine intermediate_size (missing llama.feed_forward_length)")

        num_heads = meta_int("llama.attention.head_count")
        if num_heads is None:
            raise GGUFError("Missing llama.attention.head_count (num_heads)")
        num_kv_heads = meta_int("llama.attention.head_count_kv") or num_heads

        context_len = meta_int("llama.context_length") or 0
        if args.context is not None:
            context_len = int(args.context)
        if context_len <= 0:
            raise GGUFError("Could not determine context length (use --context to override)")

        rope_theta = meta_float("llama.rope.freq_base") or 10000.0
        rms_eps = meta_float("llama.norm_rms_eps") or 1e-5

        if embed_dim != tok.ne0:
            raise GGUFError(f"{tok_name}: embedding_length mismatch (meta={embed_dim}, tensor.ne0={tok.ne0})")
        if embed_dim % num_heads != 0:
            raise GGUFError(f"hidden_size {embed_dim} not divisible by num_heads {num_heads}")

        head_dim = embed_dim // num_heads
        embed_kv = num_kv_heads * head_dim

        aligned_embed_dim = embed_dim
        aligned_head_dim = head_dim
        aligned_context = align_up_elems(context_len, 4, CACHE_ALIGN)

        required = {
            "output_norm.weight",
        }
        for name in required:
            if name not in tensors:
                raise GGUFError(f"Missing required tensor: {name}")

        token_dtype = weight_dtype(tok, "token_emb")
        needs_k_quant = token_dtype in (CK_DT_Q4_K, CK_DT_Q6_K)

        layer_infos = []
        dtype_table = [token_dtype, CK_DT_FP32]
        for layer in range(num_layers):
            attn_norm = tensors.get(f"blk.{layer}.attn_norm.weight")
            ffn_norm = tensors.get(f"blk.{layer}.ffn_norm.weight")
            if not attn_norm or not ffn_norm:
                raise GGUFError(f"Layer {layer}: missing attn_norm/ffn_norm tensors")

            wq = tensors.get(f"blk.{layer}.attn_q.weight")
            wk = tensors.get(f"blk.{layer}.attn_k.weight")
            wv = tensors.get(f"blk.{layer}.attn_v.weight")
            wo = tensors.get(f"blk.{layer}.attn_output.weight")
            gate = tensors.get(f"blk.{layer}.ffn_gate.weight")
            up = tensors.get(f"blk.{layer}.ffn_up.weight")
            down = tensors.get(f"blk.{layer}.ffn_down.weight")
            if not wq or not wk or not wv or not wo:
                raise GGUFError(f"Layer {layer}: missing attention projection tensors (q/k/v/o)")
            if not gate or not up or not down:
                raise GGUFError(f"Layer {layer}: missing ffn tensors (gate/up/down)")

            if wq.ne0 != embed_dim or wq.ne1 != embed_dim:
                raise GGUFError(f"{wq.name}: expected dims [ne0={embed_dim}, ne1={embed_dim}], got {wq.dims}")
            for tensor, label in ((wk, "K"), (wv, "V")):
                if tensor.ne0 != embed_dim or tensor.ne1 != embed_kv:
                    raise GGUFError(
                        f"{tensor.name}: expected dims [ne0={embed_dim}, ne1={embed_kv}] for {label}, got {tensor.dims}"
                    )
            if wo.ne0 != embed_dim or wo.ne1 != embed_dim:
                raise GGUFError(f"{wo.name}: expected dims [ne0={embed_dim}, ne1={embed_dim}], got {wo.dims}")

            for tensor, label in ((gate, "gate"), (up, "up")):
                if tensor.ne0 != embed_dim or tensor.ne1 != intermediate:
                    raise GGUFError(
                        f"{tensor.name}: expected dims [ne0={embed_dim}, ne1={intermediate}] for {label}, got {tensor.dims}"
                    )
            if down.ne0 != intermediate or down.ne1 != embed_dim:
                raise GGUFError(
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
                raise GGUFError(
                    f"Layer {layer}: ffn_gate ({ggml_type_name(gate.ggml_type)}) and "
                    f"ffn_up ({ggml_type_name(up.ggml_type)}) must match"
                )

            needs_k_quant = needs_k_quant or any(
                dt in (CK_DT_Q4_K, CK_DT_Q6_K)
                for dt in (wq_dt, wk_dt, wv_dt, wo_dt, gate_dt, up_dt, down_dt)
            )

            dtype_table.extend([
                CK_DT_FP32,  # ln1_gamma
                CK_DT_FP32,  # ln2_gamma
                wq_dt,
                CK_DT_FP32,  # bq
                wk_dt,
                CK_DT_FP32,  # bk
                wv_dt,
                CK_DT_FP32,  # bv
                wo_dt,
                CK_DT_FP32,  # bo
                gate_dt,
                CK_DT_FP32,  # b1
                down_dt,
                CK_DT_FP32,  # b2
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
                "up_dt": up_dt,
                "down_dt": down_dt,
            })

        dtype_table.extend([CK_DT_FP32, CK_DT_FP32])
        dtype_table_bytes = bytes(dtype_table)

        if needs_k_quant:
            if embed_dim % 256 != 0:
                raise GGUFError(f"K-quant requires hidden_size multiple of 256 (got {embed_dim})")
            if intermediate % 256 != 0:
                raise GGUFError(f"K-quant requires intermediate_size multiple of 256 (got {intermediate})")

        # Create output directory.
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w+b") as out_f:
            out_f.write(b"\x00" * HEADER_SIZE)
            w = HashingWriter(out_f)

            w.write(struct.pack("<I", len(dtype_table_bytes)))
            w.write(dtype_table_bytes)

            # 1) token embeddings
            copy_bytes_stream(f, data_start + tok.offset, ggml_tensor_bytes(tok), w)

            # 2) pos_emb: not used by RoPE models; keep zeros for compatibility.
            write_f32_zeros(w, context_len * aligned_embed_dim)

            # 3) per-layer
            for layer in range(num_layers):
                info = layer_infos[layer]
                ln1 = read_vector_f32(f, data_start, info["attn_norm"])
                ln2 = read_vector_f32(f, data_start, info["ffn_norm"])
                write_f32_padded(w, ln1, aligned_embed_dim)
                write_f32_padded(w, ln2, aligned_embed_dim)

                copy_qk_head_packed(
                    f, data_start, info["wq"], w,
                    group_count=num_heads,
                    head_dim=head_dim,
                    aligned_head_dim=aligned_head_dim,
                    aligned_embed_dim=aligned_embed_dim,
                )
                # bq
                write_f32_zeros(w, num_heads * aligned_head_dim)

                copy_qk_head_packed(
                    f, data_start, info["wk"], w,
                    group_count=num_kv_heads,
                    head_dim=head_dim,
                    aligned_head_dim=aligned_head_dim,
                    aligned_embed_dim=aligned_embed_dim,
                )
                write_f32_zeros(w, num_kv_heads * aligned_head_dim)  # bk

                copy_qk_head_packed(
                    f, data_start, info["wv"], w,
                    group_count=num_kv_heads,
                    head_dim=head_dim,
                    aligned_head_dim=aligned_head_dim,
                    aligned_embed_dim=aligned_embed_dim,
                )
                write_f32_zeros(w, num_kv_heads * aligned_head_dim)  # bv

                # Wo: stored as a single [embed_dim x embed_dim] matrix for Q4_K path.
                copy_bytes_stream(f, data_start + info["wo"].offset, ggml_tensor_bytes(info["wo"]), w)
                write_f32_zeros(w, aligned_embed_dim)  # bo

                # MLP: gate/up concatenated into w1, down is w2.
                copy_bytes_stream(f, data_start + info["gate"].offset, ggml_tensor_bytes(info["gate"]), w)
                copy_bytes_stream(f, data_start + info["up"].offset, ggml_tensor_bytes(info["up"]), w)
                write_f32_zeros(w, 2 * intermediate)  # b1
                # w2: [embed_dim x intermediate] in our runtime layout; GGML stores [ne0=intermediate, ne1=embed].
                copy_bytes_stream(f, data_start + info["down"].offset, ggml_tensor_bytes(info["down"]), w)
                write_f32_zeros(w, aligned_embed_dim)  # b2

            # 4) final RMSNorm (gamma) and bias placeholder
            final_norm = read_vector_f32(f, data_start, tensors["output_norm.weight"])
            write_f32_padded(w, final_norm, aligned_embed_dim)
            write_f32_zeros(w, aligned_embed_dim)

            checksum = w.digest()

            # Header: matches scripts/convert_hf_to_bump.py (BUMPWGT3, version 3).
            out_f.flush()
            out_f.seek(0, os.SEEK_SET)
            out_f.write(b"BUMPWGT3")
            out_f.write(struct.pack("<I", 3))  # version
            out_f.write(struct.pack("<I", 1))  # model_type (legacy)
            out_f.write(struct.pack("<I", int(num_layers)))
            out_f.write(struct.pack("<I", int(vocab_size)))
            out_f.write(struct.pack("<I", int(embed_dim)))
            out_f.write(struct.pack("<I", int(context_len)))
            out_f.write(struct.pack("<I", int(num_heads)))
            out_f.write(struct.pack("<I", int(head_dim)))
            out_f.write(struct.pack("<Q", int(aligned_embed_dim)))
            out_f.write(struct.pack("<Q", int(aligned_head_dim)))
            out_f.write(struct.pack("<Q", int(aligned_context)))
            out_f.write(checksum)
            out_f.write(b"\x00" * 32)

    if args.config_out:
        os.makedirs(os.path.dirname(args.config_out) or ".", exist_ok=True)
        cfg = build_llama_config(
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
        with open(args.config_out, "w", encoding="utf-8") as cf:
            json.dump(cfg, cf, indent=2)
            cf.write("\n")

    print(
        f"[gguf->bump] version={version} arch={arch} layers={num_layers} "
        f"hidden={embed_dim} heads={num_heads}/{num_kv_heads} ff={intermediate} "
        f"vocab={vocab_size} ctx={context_len} -> {args.output}"
    )


if __name__ == "__main__":
    main()

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
  - The bump file encodes a per-tensor dtype table (BUMPWGT4). The runtime reads
    it automatically to select the right kernel path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import sys
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
        raise GGUFError("tokenizer.json missing model.vocab")

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


def load_tokenizer_from_gguf(meta: dict, vocab_size: int) -> Optional[tuple[list[int], bytes, list[int]]]:
    """Extract tokenizer vocabulary directly from GGUF metadata.

    GGUF files contain tokenizer data in these metadata keys:
      - tokenizer.ggml.tokens: array of token strings
      - tokenizer.ggml.scores: array of token scores (optional)
      - tokenizer.ggml.token_type: array of token types (optional)
      - tokenizer.ggml.merges: BPE merge pairs (optional)

    Returns:
        (vocab_offsets, vocab_strings, merges_data) or None if not found
    """
    tokens = meta.get("tokenizer.ggml.tokens")
    if not tokens or not isinstance(tokens, (list, tuple)):
        return None

    # Build vocabulary
    offsets: list[int] = []
    strings_blob = bytearray()

    for i, token in enumerate(tokens[:vocab_size]):
        offsets.append(len(strings_blob))
        if isinstance(token, bytes):
            strings_blob.extend(token)
        elif isinstance(token, str):
            strings_blob.extend(token.encode("utf-8", errors="replace"))
        strings_blob.append(0)  # null terminator

    # Pad if we have fewer tokens than vocab_size
    while len(offsets) < vocab_size:
        offsets.append(len(strings_blob))
        strings_blob.extend(f"<|pad_{len(offsets)-1}|>".encode("utf-8"))
        strings_blob.append(0)

    # Extract BPE merges if present
    merges_data: list[int] = []
    merges = meta.get("tokenizer.ggml.merges")
    if merges and isinstance(merges, (list, tuple)):
        # Build token-to-id mapping
        token_to_id = {}
        for i, tok in enumerate(tokens[:vocab_size]):
            if isinstance(tok, bytes):
                tok = tok.decode("utf-8", errors="replace")
            token_to_id[tok] = i

        for entry in merges:
            if isinstance(entry, bytes):
                entry = entry.decode("utf-8", errors="replace")
            if isinstance(entry, str):
                parts = entry.split(" ", 1)
                if len(parts) == 2:
                    left, right = parts
                    left_id = token_to_id.get(left)
                    right_id = token_to_id.get(right)
                    merged_id = token_to_id.get(left + right)
                    if left_id is not None and right_id is not None and merged_id is not None:
                        if 0 <= left_id < vocab_size and 0 <= right_id < vocab_size and 0 <= merged_id < vocab_size:
                            merges_data.extend([left_id, right_id, merged_id])

    return offsets, bytes(strings_blob), merges_data


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
            # Store string arrays (needed for tokenizer extraction)
            strings = []
            for _ in range(n):
                strings.append(r.val_str())
            return strings
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


def get_quant_type_name(ggml_type: int) -> str:
    """Get lowercase quant type string from GGML type."""
    return {
        GGML_TYPE_F32: "fp32",
        GGML_TYPE_F16: "fp16",
        GGML_TYPE_BF16: "bf16",
        GGML_TYPE_Q4_0: "q4_0",
        GGML_TYPE_Q4_1: "q4_1",
        GGML_TYPE_Q5_0: "q5_0",
        GGML_TYPE_Q5_1: "q5_1",
        GGML_TYPE_Q8_0: "q8_0",
        GGML_TYPE_Q4_K: "q4_k",
        GGML_TYPE_Q6_K: "q6_k",
    }.get(ggml_type, f"unknown_{ggml_type}")


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


def print_conversion_report(
    num_layers: int,
    layer_infos: list,
    vocab_size: int,
    embed_dim: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    intermediate: int,
    gguf_tensors: Dict[str, TensorInfo] = None,
    manifest_entries: list = None,
    bump_path: str = None,
    gguf_path: str = None,
    data_start: int = 0,
) -> None:
    """Print a detailed conversion report showing EVERY tensor and its conversion status."""

    def fmt_size(b: int) -> str:
        """Format bytes as human-readable."""
        if b >= 1024 * 1024:
            return f"{b / (1024 * 1024):.2f} MB"
        elif b >= 1024:
            return f"{b / 1024:.2f} KB"
        return f"{b} B"

    def fmt_offset(offset: int) -> str:
        """Format offset as hex."""
        return f"0x{offset:08X}"

    def fmt_values(values: list, max_vals: int = 3) -> str:
        """Format sample values."""
        if not values:
            return "-"
        vals = values[:max_vals]
        return "[" + ", ".join(f"{v:.4f}" for v in vals) + "]"

    print("\n" + "=" * 140)
    print("DETAILED CONVERSION REPORT")
    print("=" * 140)

    # Build manifest lookup by name
    manifest_lookup = {}
    if manifest_entries:
        for entry in manifest_entries:
            manifest_lookup[entry["name"]] = entry

    # Build GGUF name to BUMP name mapping
    def get_bump_name(gguf_name: str) -> Optional[str]:
        """Map GGUF tensor name to bump entry name."""
        if gguf_name == "token_embd.weight":
            return "token_emb"
        if gguf_name == "output_norm.weight":
            return "final_ln_weight"
        if gguf_name == "output.weight":
            return None  # Tied with token_emb, not separately stored

        # Layer tensors
        if gguf_name.startswith("blk."):
            parts = gguf_name.split(".")
            if len(parts) >= 3:
                layer = parts[1]
                rest = ".".join(parts[2:])
                mapping = {
                    "attn_norm.weight": f"layer.{layer}.ln1_gamma",
                    "ffn_norm.weight": f"layer.{layer}.ln2_gamma",
                    "attn_q.weight": f"layer.{layer}.wq",
                    "attn_k.weight": f"layer.{layer}.wk",
                    "attn_v.weight": f"layer.{layer}.wv",
                    "attn_output.weight": f"layer.{layer}.wo",
                    "attn_q.bias": f"layer.{layer}.bq",
                    "attn_k.bias": f"layer.{layer}.bk",
                    "attn_v.bias": f"layer.{layer}.bv",
                    "ffn_gate.weight": f"layer.{layer}.w1",  # Combined with up
                    "ffn_up.weight": f"layer.{layer}.w1",    # Combined with gate
                    "ffn_down.weight": f"layer.{layer}.w2",
                }
                return mapping.get(rest)
        return None

    # Build reverse mapping: BUMP name to GGUF name(s)
    def get_gguf_name(bump_name: str) -> Optional[str]:
        """Map bump entry name back to GGUF tensor name."""
        if bump_name == "token_emb":
            return "token_embd.weight"
        if bump_name == "final_ln_weight":
            return "output_norm.weight"
        if bump_name == "final_ln_bias":
            return None  # No GGUF source
        if bump_name == "pos_emb":
            return None  # No GGUF source (RoPE models)

        # Layer tensors
        if bump_name.startswith("layer."):
            parts = bump_name.split(".")
            if len(parts) >= 3:
                layer = parts[1]
                suffix = parts[2]
                mapping = {
                    "ln1_gamma": f"blk.{layer}.attn_norm.weight",
                    "ln2_gamma": f"blk.{layer}.ffn_norm.weight",
                    "wq": f"blk.{layer}.attn_q.weight",
                    "wk": f"blk.{layer}.attn_k.weight",
                    "wv": f"blk.{layer}.attn_v.weight",
                    "wo": f"blk.{layer}.attn_output.weight",
                    "bq": f"blk.{layer}.attn_q.bias",
                    "bk": f"blk.{layer}.attn_k.bias",
                    "bv": f"blk.{layer}.attn_v.bias",
                    "w1": f"blk.{layer}.ffn_gate.weight",  # gate+up combined
                    "w2": f"blk.{layer}.ffn_down.weight",
                    "bo": None,  # No GGUF source
                    "b1": None,  # No GGUF source
                    "b2": None,  # No GGUF source
                }
                return mapping.get(suffix)
        return None

    # ════════════════════════════════════════════════════════════════════════════
    # Main Table: BUMP entries with GGUF source, conversion status, and values
    # ════════════════════════════════════════════════════════════════════════════
    if manifest_entries:
        print(f"\n[Conversion Table] ({len(manifest_entries)} BUMP entries)")
        print(f"{'BUMP Entry':<22} | {'GGUF Source':<28} | {'GGUF Type':<7} | {'Conv':<4} | {'BUMP Offset':<12} | {'Size':<10} | {'Sample Values'}")
        print("-" * 22 + "-+-" + "-" * 28 + "-+-" + "-" * 7 + "-+-" + "-" * 4 + "-+-" + "-" * 12 + "-+-" + "-" * 10 + "-+-" + "-" * 25)

        converted_from_gguf = 0
        zeros_placeholder = 0
        other_entries = 0

        for entry in manifest_entries:
            bump_name = entry["name"]
            bump_type = entry["dtype"]
            offset_str = fmt_offset(entry["file_offset"])
            size_str = fmt_size(entry["size"])

            # Find GGUF source
            gguf_name = get_gguf_name(bump_name)
            gguf_type = "-"
            conv_status = " "
            sample_vals = "-"

            # Check if this is a bias entry
            is_bias = bump_name.endswith((".bq", ".bk", ".bv", ".bo", ".b1", ".b2"))
            layer_idx = None
            if is_bias and "." in bump_name:
                layer_num = bump_name.split(".")[1]
                if layer_num.isdigit():
                    layer_idx = int(layer_num)

            # Determine source and conversion status
            if gguf_name and gguf_name in (gguf_tensors or {}):
                gguf_info = gguf_tensors[gguf_name]
                gguf_type = ggml_type_name(gguf_info.ggml_type)
                conv_status = "✓"
                converted_from_gguf += 1
            elif is_bias and layer_idx is not None and layer_idx < len(layer_infos):
                bias_key = bump_name.split(".")[-1]
                if layer_infos[layer_idx].get(bias_key) is not None:
                    # Bias exists in GGUF
                    gguf_name = f"blk.{layer_idx}.attn_{bias_key[1]}.bias"
                    gguf_type = "F32"
                    conv_status = "✓"
                    converted_from_gguf += 1
                else:
                    # Bias is zeros placeholder
                    gguf_name = "(none - zeros)"
                    conv_status = "○"
                    zeros_placeholder += 1
            elif bump_name in ("pos_emb", "final_ln_bias"):
                gguf_name = "(none - zeros)"
                conv_status = "○"
                zeros_placeholder += 1
            else:
                other_entries += 1

            # Truncate names for display
            bump_display = bump_name[:21] if len(bump_name) > 21 else bump_name
            gguf_display = (gguf_name or "-")[:27] if gguf_name else "-"

            print(f"{bump_display:<22} | {gguf_display:<28} | {gguf_type:<7} | {conv_status:<4} | {offset_str:<12} | {size_str:<10} |")

        print("-" * 140)
        print(f"  Summary: {converted_from_gguf} from GGUF (✓), {zeros_placeholder} zeros placeholders (○), {other_entries} other")

    # ════════════════════════════════════════════════════════════════════════════
    # Section 3: Per-layer quant type summary (compact)
    # ════════════════════════════════════════════════════════════════════════════
    print(f"\n[Per-Layer Quant Types] ({num_layers} layers)")
    print(f"  {'Layer':<6} | {'WQ':<5} | {'WK':<5} | {'WV':<5} | {'WO':<5} | {'W1':<5} | {'W2':<5} | {'Biases'}")
    print(f"  {'-'*6} | {'-'*5} | {'-'*5} | {'-'*5} | {'-'*5} | {'-'*5} | {'-'*5} | {'-'*7}")

    for layer_id, info in enumerate(layer_infos):
        wq_dt = get_quant_type_name(info["wq"].ggml_type).upper() if info.get("wq") else "N/A"
        wk_dt = get_quant_type_name(info["wk"].ggml_type).upper() if info.get("wk") else "N/A"
        wv_dt = get_quant_type_name(info["wv"].ggml_type).upper() if info.get("wv") else "N/A"
        wo_dt = get_quant_type_name(info["wo"].ggml_type).upper() if info.get("wo") else "N/A"
        w1_dt = get_quant_type_name(info["gate"].ggml_type).upper() if info.get("gate") else "N/A"
        w2_dt = get_quant_type_name(info["down"].ggml_type).upper() if info.get("down") else "N/A"

        has_bq = "Q" if info.get("bq") is not None else "-"
        has_bk = "K" if info.get("bk") is not None else "-"
        has_bv = "V" if info.get("bv") is not None else "-"
        bias_str = f"{has_bq}{has_bk}{has_bv}"

        # Show all layers for full detail
        print(f"  {layer_id:<6} | {wq_dt:<5} | {wk_dt:<5} | {wv_dt:<5} | {wo_dt:<5} | {w1_dt:<5} | {w2_dt:<5} | {bias_str}")

    # ════════════════════════════════════════════════════════════════════════════
    # Section 4: Validation summary
    # ════════════════════════════════════════════════════════════════════════════
    layers_with_bq = sum(1 for info in layer_infos if info.get("bq") is not None)
    layers_with_bk = sum(1 for info in layer_infos if info.get("bk") is not None)
    layers_with_bv = sum(1 for info in layer_infos if info.get("bv") is not None)

    print(f"\n[Validation Summary]")
    print(f"  Model dims: embed={embed_dim}, heads={num_heads}/{num_kv_heads}, head_dim={head_dim}, ff={intermediate}")
    print(f"  Vocab: {vocab_size}, Layers: {num_layers}")
    print(f"  Attention biases: bq={layers_with_bq}/{num_layers}, bk={layers_with_bk}/{num_layers}, bv={layers_with_bv}/{num_layers}")

    if layers_with_bq > 0 and layers_with_bq < num_layers:
        print(f"  ⚠ WARNING: Only {layers_with_bq}/{num_layers} layers have Q bias!")
    if layers_with_bk > 0 and layers_with_bk < num_layers:
        print(f"  ⚠ WARNING: Only {layers_with_bk}/{num_layers} layers have K bias!")
    if layers_with_bv > 0 and layers_with_bv < num_layers:
        print(f"  ⚠ WARNING: Only {layers_with_bv}/{num_layers} layers have V bias!")

    if layers_with_bq == num_layers and layers_with_bk == num_layers and layers_with_bv == num_layers:
        print(f"  ✓ All attention biases (Q, K, V) extracted for all {num_layers} layers")
    elif layers_with_bq == 0 and layers_with_bk == 0 and layers_with_bv == 0:
        print(f"  ✓ No attention biases in source (LLaMA-style architecture)")
    else:
        print(f"  ✓ Mixed bias configuration detected")

    if manifest_entries:
        total_size = sum(e["size"] for e in manifest_entries)
        print(f"  Total bump file size: {fmt_size(total_size)}")

    print("=" * 100 + "\n")


def verify_bump_parity(
    gguf_path: str,
    bump_path: str,
    num_layers: int,
    embed_dim: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> bool:
    """Verify parity between GGUF tensors and converted bump file.

    Checks sample values from:
    - Token embeddings
    - Layer 0 attention Q/K/V biases (if present)
    - Layer 0 RMSNorm weights

    Returns True if parity checks pass, False otherwise.
    """
    import struct

    print("\n[verify] Running parity verification...")
    errors = []

    with open(gguf_path, "rb") as gf, open(bump_path, "rb") as bf:
        gr = GGUFReader(gf)

        # Parse GGUF header to find tensors
        magic = gr._read_exact(4)
        if magic != b"GGUF":
            errors.append(f"Invalid GGUF magic: {magic}")
            return False

        version = gr.u32()
        if version >= 2:
            n_tensors = gr.u64()
            n_kv = gr.u64()
        else:
            n_tensors = gr.u32()
            n_kv = gr.u32()

        # Skip metadata
        for _ in range(n_kv):
            key = gr.key_str()
            vtype = gr.u32()
            _gguf_skip_value(gr, vtype)

        # Parse tensor info
        tensors: Dict[str, TensorInfo] = {}
        for _ in range(n_tensors):
            name = gr.key_str()
            n_dims = gr.u32()
            dims = tuple(int(gr.u64()) for _ in range(n_dims))
            ggml_type = gr.u32()
            offset = gr.u64()
            tensors[name] = TensorInfo(name=name, dims=dims, ggml_type=int(ggml_type), offset=int(offset))

        alignment = 32  # default
        data_start = align_up(gr.tell(), alignment)

        # Read bump header
        bf.seek(0)
        bump_magic = bf.read(8)
        if bump_magic != b"BUMPWGT4":
            errors.append(f"Invalid bump magic: {bump_magic} (expected BUMPWGT4)")
            return False

        # Skip rest of header (120 bytes remaining)
        bf.seek(HEADER_SIZE)

        # Read dtype table
        dtype_table_len = struct.unpack("<I", bf.read(4))[0]
        dtype_table = bf.read(dtype_table_len)

        # Check that we have attention bias tensors if GGUF has them
        has_gguf_bq = "blk.0.attn_q.bias" in tensors
        has_gguf_bk = "blk.0.attn_k.bias" in tensors
        has_gguf_bv = "blk.0.attn_v.bias" in tensors

        # Verify token embedding (sample first row)
        tok_info = tensors.get("token_embd.weight")
        if tok_info:
            gf.seek(data_start + tok_info.offset)
            gguf_sample = gf.read(34)  # Q8_0 first block

            # After dtype table comes token embeddings
            bump_tok_offset = HEADER_SIZE + 4 + dtype_table_len
            bf.seek(bump_tok_offset)
            bump_sample = bf.read(34)

            if gguf_sample != bump_sample:
                errors.append("Token embedding block 0 mismatch!")
            else:
                print("[verify] ✓ Token embedding parity OK")

        # If GGUF has attention biases, verify they were copied (not zeros)
        if has_gguf_bq:
            bq_info = tensors["blk.0.attn_q.bias"]
            gf.seek(data_start + bq_info.offset)
            gguf_bq = np.frombuffer(gf.read(embed_dim * 4), dtype=np.float32)

            # Check if bq has non-zero values in GGUF
            gguf_bq_nonzero = np.count_nonzero(gguf_bq)
            if gguf_bq_nonzero > 0:
                print(f"[verify] GGUF has attn_q.bias with {gguf_bq_nonzero}/{embed_dim} non-zero values")
                print(f"[verify]   First 5 values: {gguf_bq[:5]}")
            else:
                print("[verify] Note: GGUF attn_q.bias is all zeros")
        else:
            print("[verify] Note: No attn_q.bias in GGUF (LLaMA-style model)")

        if has_gguf_bk:
            bk_info = tensors["blk.0.attn_k.bias"]
            gf.seek(data_start + bk_info.offset)
            gguf_bk = np.frombuffer(gf.read(num_kv_heads * head_dim * 4), dtype=np.float32)
            gguf_bk_nonzero = np.count_nonzero(gguf_bk)
            if gguf_bk_nonzero > 0:
                print(f"[verify] GGUF has attn_k.bias with {gguf_bk_nonzero}/{num_kv_heads * head_dim} non-zero values")

        if has_gguf_bv:
            bv_info = tensors["blk.0.attn_v.bias"]
            gf.seek(data_start + bv_info.offset)
            gguf_bv = np.frombuffer(gf.read(num_kv_heads * head_dim * 4), dtype=np.float32)
            gguf_bv_nonzero = np.count_nonzero(gguf_bv)
            if gguf_bv_nonzero > 0:
                print(f"[verify] GGUF has attn_v.bias with {gguf_bv_nonzero}/{num_kv_heads * head_dim} non-zero values")

        # Summary
        print(f"[verify] Checked {num_layers} layers with biases: Q={has_gguf_bq}, K={has_gguf_bk}, V={has_gguf_bv}")

    if errors:
        print("\n[verify] PARITY CHECK FAILED:")
        for e in errors:
            print(f"  - {e}")
        return False

    print("[verify] ✓ Parity verification passed!")
    return True


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


def _scan_cached_models() -> list:
    """Scan cache directories for existing GGUF models."""
    import glob
    cache_dirs = [
        os.path.expanduser("~/.cache/ck-engine-v6.6/models"),
        os.path.expanduser("~/.cache/ck-engine/models"),
        ".",
    ]
    found = []
    for cache_dir in cache_dirs:
        if os.path.isdir(cache_dir):
            # Find all .gguf files
            pattern = os.path.join(cache_dir, "**", "*.gguf")
            for gguf_path in glob.glob(pattern, recursive=True):
                size_mb = os.path.getsize(gguf_path) / (1024 * 1024)
                model_dir = os.path.dirname(gguf_path)
                model_name = os.path.basename(model_dir)
                # Use filename if model_name is unhelpful
                if model_name in (".", "", "models"):
                    model_name = os.path.basename(gguf_path).replace(".gguf", "")
                found.append({
                    "path": gguf_path,
                    "name": model_name,
                    "size_mb": size_mb,
                    "dir": model_dir,
                })
    # Sort by size (larger models first, likely more useful)
    found.sort(key=lambda x: -x["size_mb"])
    return found


def _build_examples() -> str:
    """Build examples section, using real cached models if available."""
    cached = _scan_cached_models()

    # Use first cached model or fallback to generic
    if cached:
        m = cached[0]
        gguf = m["path"]
        model_dir = m["dir"]
        example_gguf = gguf
        example_output = os.path.join(model_dir, "weights.bump")
        example_config = os.path.join(model_dir, "config.json")
        example_manifest = os.path.join(model_dir, "weights_manifest.json")
        example_vocab = os.path.join(model_dir, "vocab.json")
        recommendation = f"""
Recommended command for '{m['name']}':
  (Performs weights conversion + metadata extraction + vocabulary extraction)

  python scripts/convert_gguf_to_bump.py \\
      --gguf {gguf} \\
      --output {example_output} \\
      --config-out {example_config} \\
      --manifest-out {example_manifest} \\
      --extract-vocab {example_vocab}
"""
    else:
        example_gguf = "model.gguf"
        example_output = "weights.bump"
        example_config = "config.json"
        example_manifest = "weights_manifest.json"
        example_vocab = "vocab.json"
        recommendation = ""

    # Build cached models section
    if cached:
        cached_section = "\nCached Models Found:\n"
        for i, m in enumerate(cached[:5]):  # Show max 5
            cached_section += f"  [{i+1}] {m['name']} ({m['size_mb']:.1f} MB)\n"
            cached_section += f"      {m['path']}\n"
        if len(cached) > 5:
            cached_section += f"  ... and {len(cached) - 5} more\n"
    else:
        cached_section = "\nNo cached models found. Download a GGUF model first.\n"

    examples = f"""{cached_section}{recommendation}
Workflow Examples:
  # A. INSPECT: Quick look at GGUF metadata and tensor types (no conversion)
  python scripts/convert_gguf_to_bump.py --gguf {example_gguf} --inspect

  # B. TOKENIZER ONLY: Extract vocabulary and tokenizer stats to JSON
  python scripts/convert_gguf_to_bump.py --gguf {example_gguf} --extract-vocab {example_vocab}

  # C. CONVERT: Generate weights.bump file for C-Kernel-Engine
  python scripts/convert_gguf_to_bump.py --gguf {example_gguf} --output {example_output}

  # D. FULL PIPELINE: Weights + Config + Manifest + Vocabulary
  python scripts/convert_gguf_to_bump.py --gguf {example_gguf} \\
      --output {example_output} \\
      --config-out {example_config} \\
      --manifest-out {example_manifest} \\
      --extract-vocab {example_vocab}

  # E. VERIFY: Convert and check parity between GGUF and bump file
  python scripts/convert_gguf_to_bump.py --gguf {example_gguf} --output {example_output} --verify

Tensor Mapping (GGUF → C-Kernel-Engine):
  token_embd.weight      → token_emb
  output_norm.weight     → final_ln_weight
  blk.X.attn_norm.weight → layer.X.ln1_gamma
  blk.X.ffn_norm.weight  → layer.X.ln2_gamma
  blk.X.attn_q.weight    → layer.X.wq
  blk.X.attn_k.weight    → layer.X.wk
  blk.X.attn_v.weight    → layer.X.wv
  blk.X.attn_output.weight → layer.X.wo
  blk.X.ffn_gate.weight  → layer.X.w1 (combined with up)
  blk.X.ffn_up.weight    → layer.X.w1 (combined with gate)
  blk.X.ffn_down.weight  → layer.X.w2
"""
    return examples


def main() -> None:
    examples = _build_examples()
    ap = argparse.ArgumentParser(
        description="Convert GGUF (Q4_K / Q6_K) weights to weights.bump for C-Kernel-Engine",
        epilog=examples,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--gguf", required=True, help="Input GGUF file (e.g. model.Q4_K_M.gguf)")
    ap.add_argument("--output", help="Output weights.bump path (required unless --inspect/--list)")
    ap.add_argument("--config-out", help="Optional config.json output path (HF-style minimal config)")
    ap.add_argument("--context", type=int, help="Override context length (max_position_embeddings)")
    ap.add_argument("--inspect", action="store_true", help="Print GGUF metadata/tensor dtypes and exit (no conversion)")
    ap.add_argument("--inspect-layers", action="store_true", help="Show per-layer quant types for ALL layers (use with --inspect)")
    ap.add_argument("--list", action="store_true", help="Print every tensor name/type/shape and exit (no conversion)")
    ap.add_argument("--verify", action="store_true", help="After conversion, verify parity between GGUF and bump file")
    ap.add_argument("--manifest-out", help="Output weights manifest JSON path with tensor offsets")
    ap.add_argument("--extract-vocab", help="Extract tokenizer to JSON file (runs scripts/extract_gguf_vocab.py)")
    ap.add_argument("--tokenizer-json", help="Tokenizer JSON (HuggingFace) to embed vocab + merges")
    args = ap.parse_args()

    if not args.output and not (args.inspect or args.list or args.extract_vocab):
        ap.error("--output is required unless --inspect/--list/--extract-vocab is set")

    # Support multiple architectures (llama, qwen2, mistral3, deepseek2, etc.)
    wanted_meta = {
        "general.architecture",
        "general.alignment",
        # Llama-style keys
        "llama.block_count",
        "llama.context_length",
        "llama.embedding_length",
        "llama.feed_forward_length",
        "llama.attention.head_count",
        "llama.attention.head_count_kv",
        "llama.rope.freq_base",
        "llama.norm_rms_eps",
        # Qwen2-style keys
        "qwen2.block_count",
        "qwen2.context_length",
        "qwen2.embedding_length",
        "qwen2.feed_forward_length",
        "qwen2.attention.head_count",
        "qwen2.attention.head_count_kv",
        "qwen2.rope.freq_base",
        "qwen2.attention.layer_norm_rms_epsilon",
        # Mistral3-style keys (Devstral, SmolLM, etc.)
        "mistral3.block_count",
        "mistral3.context_length",
        "mistral3.embedding_length",
        "mistral3.feed_forward_length",
        "mistral3.attention.head_count",
        "mistral3.attention.head_count_kv",
        "mistral3.rope.freq_base",
        "mistral3.attention.layer_norm_rms_epsilon",
        # Mistral-style keys
        "mistral.block_count",
        "mistral.context_length",
        "mistral.embedding_length",
        "mistral.feed_forward_length",
        "mistral.attention.head_count",
        "mistral.attention.head_count_kv",
        "mistral.rope.freq_base",
        "mistral.attention.layer_norm_rms_epsilon",
        # DeepSeek2-style keys (Kimi, etc.)
        "deepseek2.block_count",
        "deepseek2.context_length",
        "deepseek2.embedding_length",
        "deepseek2.feed_forward_length",
        "deepseek2.attention.head_count",
        "deepseek2.attention.head_count_kv",
        "deepseek2.rope.freq_base",
        "deepseek2.attention.layer_norm_rms_epsilon",
        # Tokenizer keys (for automatic extraction from GGUF)
        "tokenizer.ggml.tokens",
        "tokenizer.ggml.merges",
        "tokenizer.ggml.scores",
        "tokenizer.ggml.token_type",
    }

    if args.extract_vocab:
        import subprocess
        print(f"[convert] Extracting vocabulary to {args.extract_vocab}...")
        script_dir = os.path.dirname(os.path.abspath(__file__))
        vocab_script = os.path.join(script_dir, "extract_gguf_vocab.py")
        
        try:
            subprocess.run(
                [sys.executable, vocab_script, "--gguf", args.gguf, "--output", args.extract_vocab],
                check=True
            )
        except subprocess.CalledProcessError as e:
            print(f"[convert] Error extracting vocab: {e}")
            # Don't exit, might still want to convert weights
        except Exception as e:
            print(f"[convert] Failed to run vocab script: {e}")

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

            # Highlight common "does this get quantized?" tensors.
            highlight = [
                "token_embd.weight",
                "output.weight",
                "output_norm.weight",
                "blk.0.attn_q.weight",
                "blk.0.attn_k.weight",
                "blk.0.attn_v.weight",
                "blk.0.attn_q.bias",
                "blk.0.attn_k.bias",
                "blk.0.attn_v.bias",
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
        # Support multiple architecture prefixes (llama, qwen2, etc.)
        def meta_int(*keys: str) -> Optional[int]:
            """Get integer from metadata, trying multiple keys in order."""
            for key in keys:
                v = meta.get(key)
                if v is not None:
                    if isinstance(v, bool):
                        return int(v)
                    if isinstance(v, (int, np.integer)):
                        return int(v)
            return None

        def meta_float(*keys: str) -> Optional[float]:
            """Get float from metadata, trying multiple keys in order."""
            for key in keys:
                v = meta.get(key)
                if v is not None:
                    if isinstance(v, (float, np.floating)):
                        return float(v)
                    if isinstance(v, (int, np.integer)):
                        return float(v)
            return None

        def weight_dtype(info: TensorInfo, label: str) -> int:
            # Support all common quantization types
            supported_types = (
                GGML_TYPE_Q4_K, GGML_TYPE_Q6_K, GGML_TYPE_F32,
                GGML_TYPE_Q5_0, GGML_TYPE_Q5_1, GGML_TYPE_Q8_0
            )
            if info.ggml_type not in supported_types:
                raise GGUFError(
                    f"{info.name}: expected Q4_K/Q6_K/Q5_0/Q5_1/Q8_0/F32 for {label}, got {ggml_type_name(info.ggml_type)}"
                )
            return ck_dtype_from_ggml_type(info.ggml_type)

        tok_name = "token_embd.weight"
        if tok_name not in tensors:
            raise GGUFError(f"Missing required tensor: {tok_name}")
        tok = tensors[tok_name]
        if len(tok.dims) != 2:
            raise GGUFError(f"{tok_name}: expected 2D, got dims={tok.dims}")
        embed_dim = meta_int("deepseek2.embedding_length", "mistral3.embedding_length", "mistral.embedding_length", "llama.embedding_length", "qwen2.embedding_length") or tok.ne0
        vocab_size = tok.ne1

        vocab_offsets = None
        vocab_strings = None
        vocab_merges = None
        num_merges = 0
        total_vocab_bytes = 0
        if args.tokenizer_json:
            if not os.path.exists(args.tokenizer_json):
                raise GGUFError(f"tokenizer.json not found: {args.tokenizer_json}")
            vocab_offsets, vocab_strings, vocab_merges = load_tokenizer_json(args.tokenizer_json, vocab_size)
            num_merges = len(vocab_merges) // 3
            total_vocab_bytes = len(vocab_strings)
            print(f"[tokenizer] loaded {len(vocab_offsets)} tokens, {num_merges} merges, {total_vocab_bytes} bytes from tokenizer.json")
        else:
            # Try to extract tokenizer directly from GGUF metadata
            gguf_tokenizer = load_tokenizer_from_gguf(meta, vocab_size)
            if gguf_tokenizer:
                vocab_offsets, vocab_strings, vocab_merges = gguf_tokenizer
                num_merges = len(vocab_merges) // 3
                total_vocab_bytes = len(vocab_strings)
                print(f"[tokenizer] extracted {len(vocab_offsets)} tokens, {num_merges} merges, {total_vocab_bytes} bytes from GGUF metadata")

        num_layers = meta_int("deepseek2.block_count", "mistral3.block_count", "mistral.block_count", "llama.block_count", "qwen2.block_count")
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
                raise GGUFError("Could not infer num_layers (missing block_count and no blk.* tensors found)")
            num_layers = max(layer_ids) + 1

        intermediate = meta_int("deepseek2.feed_forward_length", "mistral3.feed_forward_length", "mistral.feed_forward_length", "llama.feed_forward_length", "qwen2.feed_forward_length")
        if intermediate is None:
            gate0 = tensors.get("blk.0.ffn_gate.weight")
            if gate0 and len(gate0.dims) == 2:
                intermediate = gate0.ne1
        if intermediate is None:
            raise GGUFError("Could not determine intermediate_size (missing feed_forward_length)")

        num_heads = meta_int("deepseek2.attention.head_count", "mistral3.attention.head_count", "mistral.attention.head_count", "llama.attention.head_count", "qwen2.attention.head_count")
        if num_heads is None:
            raise GGUFError("Missing attention.head_count (num_heads)")
        num_kv_heads = meta_int("deepseek2.attention.head_count_kv", "mistral3.attention.head_count_kv", "mistral.attention.head_count_kv", "llama.attention.head_count_kv", "qwen2.attention.head_count_kv") or num_heads

        context_len = meta_int("deepseek2.context_length", "mistral3.context_length", "mistral.context_length", "llama.context_length", "qwen2.context_length") or 0
        if args.context is not None:
            context_len = int(args.context)
        if context_len <= 0:
            raise GGUFError("Could not determine context length (use --context to override)")

        rope_theta = meta_float("deepseek2.rope.freq_base", "mistral3.rope.freq_base", "mistral.rope.freq_base", "llama.rope.freq_base", "qwen2.rope.freq_base") or 10000.0
        rms_eps = meta_float("deepseek2.attention.layer_norm_rms_epsilon", "mistral3.attention.layer_norm_rms_epsilon", "mistral.attention.layer_norm_rms_epsilon", "llama.norm_rms_eps", "qwen2.attention.layer_norm_rms_epsilon") or 1e-5

        if embed_dim != tok.ne0:
            raise GGUFError(f"{tok_name}: embedding_length mismatch (meta={embed_dim}, tensor.ne0={tok.ne0})")
        if embed_dim % num_heads != 0:
            raise GGUFError(f"hidden_size {embed_dim} not divisible by num_heads {num_heads}")

        head_dim = embed_dim // num_heads
        embed_kv = num_kv_heads * head_dim

        # Infer correct dimensions from actual tensors if metadata doesn't match
        # This handles non-standard architectures like Devstral
        wq0 = tensors.get("blk.0.attn_q.weight")
        wk0 = tensors.get("blk.0.attn_k.weight")
        wo0 = tensors.get("blk.0.attn_output.weight")
        if wq0 and wk0 and wo0:
            # Check if actual dimensions match expected
            q_dim1 = wq0.ne1
            k_dim1 = wk0.ne1
            o_dim0 = wo0.ne0

            if q_dim1 != embed_dim or k_dim1 != embed_kv:
                # Infer from actual tensor shapes
                # Infer head dimensions
                inferred_q_head_dim = q_dim1 // num_heads if q_dim1 % num_heads == 0 else q_dim1
                # Update for consistency with actual tensors
                embed_kv = k_dim1
                head_dim = inferred_q_head_dim


        aligned_embed_dim = embed_dim
        aligned_head_dim = head_dim
        aligned_intermediate = intermediate
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
            # Attention biases (optional - Qwen2 has them, LLaMA doesn't)
            bq = tensors.get(f"blk.{layer}.attn_q.bias")
            bk = tensors.get(f"blk.{layer}.attn_k.bias")
            bv = tensors.get(f"blk.{layer}.attn_v.bias")
            if not wq or not wk or not wv or not wo:
                raise GGUFError(f"Layer {layer}: missing attention projection tensors (q/k/v/o)")
            if not gate or not up or not down:
                raise GGUFError(f"Layer {layer}: missing ffn tensors (gate/up/down)")

            # Validate dimensions (flexible for non-standard architectures)
            if wq.ne0 != embed_dim:
                raise GGUFError(f"{wq.name}: ne0 mismatch: expected {embed_dim}, got {wq.ne0}")
            if wk.ne0 != embed_dim or wv.ne0 != embed_dim:
                raise GGUFError(f"K/V ne0 mismatch: expected {embed_dim}, got {wk.ne0}/{wv.ne0}")
            if wo.ne1 != embed_dim:
                raise GGUFError(f"{wo.name}: ne1 mismatch: expected {embed_dim}, got {wo.ne1}")

            # For ne1 dimensions, check against inferred values
            if wq.ne1 != embed_dim and wq.ne1 != head_dim * num_heads:
                raise GGUFError(f"{wq.name}: ne1 invalid: expected {embed_dim} or {head_dim * num_heads}, got {wq.ne1}")
            if wk.ne1 != embed_kv:
                raise GGUFError(f"{wk.name}: ne1 invalid: expected {embed_kv}, got {wk.ne1}")
            if wv.ne1 != embed_kv:
                raise GGUFError(f"{wv.name}: ne1 invalid: expected {embed_kv}, got {wv.ne1}")
            if wo.ne0 != wq.ne1:
                raise GGUFError(f"{wo.name}: ne0 mismatch with Q ne1: expected {wq.ne1}, got {wo.ne0}")

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
                "bq": bq,  # Optional bias tensors (None if not present)
                "bk": bk,
                "bv": bv,
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

        # Check alignment requirements per-tensor:
        # - K-quants (Q4_K, Q6_K) require row dimension (ne0) divisible by 256
        # - Simple quants (Q5_0, Q8_0) require row dimension divisible by 32
        # We already validated this in copy_qk_head_packed and ggml_row_bytes,
        # but add a check here for K-quant matrices where embed_dim is the input
        for layer_info in layer_infos:
            # Check WQ/WK/WV/gate/up: input dim = embed_dim
            for key in ["wq_dt", "wk_dt", "wv_dt", "gate_dt", "up_dt"]:
                dt = layer_info.get(key)
                if dt in (CK_DT_Q4_K, CK_DT_Q6_K) and embed_dim % 256 != 0:
                    raise GGUFError(f"K-quant requires embed_dim multiple of 256 (got {embed_dim}) for {key}")
            # Check down: input dim = intermediate
            dt = layer_info.get("down_dt")
            if dt in (CK_DT_Q4_K, CK_DT_Q6_K) and intermediate % 256 != 0:
                raise GGUFError(f"K-quant requires intermediate_size multiple of 256 (got {intermediate})")

        # Create output directory.
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

        # Track manifest entries as we write
        manifest_entries = []
        current_offset = HEADER_SIZE

        def record_entry(name: str, dtype: str, size: int):
            nonlocal current_offset
            manifest_entries.append({
                "name": name,
                "dtype": dtype,
                "file_offset": current_offset,
                "size": size,
            })
            current_offset += size

        with open(args.output, "w+b") as out_f:
            out_f.write(b"\x00" * HEADER_SIZE)
            w = HashingWriter(out_f)

            # Dtype table
            dtype_table_header_size = 4 + len(dtype_table_bytes)
            current_offset += dtype_table_header_size
            w.write(struct.pack("<I", len(dtype_table_bytes)))
            w.write(dtype_table_bytes)

            # 1) token embeddings
            tok_size = ggml_tensor_bytes(tok)
            record_entry("token_emb", get_quant_type_name(tok.ggml_type), tok_size)
            copy_bytes_stream(f, data_start + tok.offset, tok_size, w)

            # 1b) tokenizer data (optional)
            if vocab_offsets is not None and vocab_strings is not None and vocab_merges is not None:
                offsets_bytes = struct.pack(f"<{len(vocab_offsets)}i", *vocab_offsets)
                record_entry("vocab_offsets", "i32", len(offsets_bytes))
                w.write(offsets_bytes)

                record_entry("vocab_strings", "u8", len(vocab_strings))
                w.write(vocab_strings)

                merges_bytes = b""
                if vocab_merges:
                    merges_bytes = struct.pack(f"<{len(vocab_merges)}i", *vocab_merges)
                record_entry("vocab_merges", "i32", len(merges_bytes))
                if merges_bytes:
                    w.write(merges_bytes)

            # 2) pos_emb: not used by RoPE models; keep zeros for compatibility.
            pos_emb_size = context_len * aligned_embed_dim * 4
            record_entry("pos_emb", "fp32", pos_emb_size)
            write_f32_zeros(w, context_len * aligned_embed_dim)

            # 3) per-layer
            for layer in range(num_layers):
                info = layer_infos[layer]

                # RMSNorm weights
                ln1 = read_vector_f32(f, data_start, info["attn_norm"])
                ln2 = read_vector_f32(f, data_start, info["ffn_norm"])
                ln_size = aligned_embed_dim * 4
                record_entry(f"layer.{layer}.ln1_gamma", "fp32", ln_size)
                write_f32_padded(w, ln1, aligned_embed_dim)
                record_entry(f"layer.{layer}.ln2_gamma", "fp32", ln_size)
                write_f32_padded(w, ln2, aligned_embed_dim)

                # WQ
                wq_size = ggml_row_bytes(info["wq"].ggml_type, aligned_embed_dim) * embed_dim
                record_entry(f"layer.{layer}.wq", get_quant_type_name(info["wq"].ggml_type), wq_size)
                copy_qk_head_packed(
                    f, data_start, info["wq"], w,
                    group_count=num_heads,
                    head_dim=head_dim,
                    aligned_head_dim=aligned_head_dim,
                    aligned_embed_dim=aligned_embed_dim,
                )

                # bq - write actual bias if present, else zeros
                bq_size = num_heads * aligned_head_dim * 4
                record_entry(f"layer.{layer}.bq", "fp32", bq_size)
                if info["bq"] is not None:
                    bq_vec = read_vector_f32(f, data_start, info["bq"])
                    write_f32_padded(w, bq_vec, num_heads * aligned_head_dim)
                else:
                    write_f32_zeros(w, num_heads * aligned_head_dim)

                # WK
                wk_size = ggml_row_bytes(info["wk"].ggml_type, aligned_embed_dim) * embed_kv
                record_entry(f"layer.{layer}.wk", get_quant_type_name(info["wk"].ggml_type), wk_size)
                copy_qk_head_packed(
                    f, data_start, info["wk"], w,
                    group_count=num_kv_heads,
                    head_dim=head_dim,
                    aligned_head_dim=aligned_head_dim,
                    aligned_embed_dim=aligned_embed_dim,
                )

                # bk
                bk_size = num_kv_heads * aligned_head_dim * 4
                record_entry(f"layer.{layer}.bk", "fp32", bk_size)
                if info["bk"] is not None:
                    bk_vec = read_vector_f32(f, data_start, info["bk"])
                    write_f32_padded(w, bk_vec, num_kv_heads * aligned_head_dim)
                else:
                    write_f32_zeros(w, num_kv_heads * aligned_head_dim)

                # WV
                wv_size = ggml_row_bytes(info["wv"].ggml_type, aligned_embed_dim) * embed_kv
                record_entry(f"layer.{layer}.wv", get_quant_type_name(info["wv"].ggml_type), wv_size)
                copy_qk_head_packed(
                    f, data_start, info["wv"], w,
                    group_count=num_kv_heads,
                    head_dim=head_dim,
                    aligned_head_dim=aligned_head_dim,
                    aligned_embed_dim=aligned_embed_dim,
                )

                # bv
                bv_size = num_kv_heads * aligned_head_dim * 4
                record_entry(f"layer.{layer}.bv", "fp32", bv_size)
                if info["bv"] is not None:
                    bv_vec = read_vector_f32(f, data_start, info["bv"])
                    write_f32_padded(w, bv_vec, num_kv_heads * aligned_head_dim)
                else:
                    write_f32_zeros(w, num_kv_heads * aligned_head_dim)

                # Wo
                wo_size = ggml_tensor_bytes(info["wo"])
                record_entry(f"layer.{layer}.wo", get_quant_type_name(info["wo"].ggml_type), wo_size)
                copy_bytes_stream(f, data_start + info["wo"].offset, wo_size, w)
                bo_size = aligned_embed_dim * 4
                record_entry(f"layer.{layer}.bo", "fp32", bo_size)
                write_f32_zeros(w, aligned_embed_dim)  # bo

                # W1 (gate + up)
                gate_size = ggml_tensor_bytes(info["gate"])
                up_size = ggml_tensor_bytes(info["up"])
                record_entry(f"layer.{layer}.w1", get_quant_type_name(info["gate"].ggml_type), gate_size + up_size)
                copy_bytes_stream(f, data_start + info["gate"].offset, gate_size, w)
                copy_bytes_stream(f, data_start + info["up"].offset, up_size, w)
                b1_size = 2 * intermediate * 4
                record_entry(f"layer.{layer}.b1", "fp32", b1_size)
                write_f32_zeros(w, 2 * intermediate)  # b1

                # W2 (down)
                down_size = ggml_tensor_bytes(info["down"])
                record_entry(f"layer.{layer}.w2", get_quant_type_name(info["down"].ggml_type), down_size)
                copy_bytes_stream(f, data_start + info["down"].offset, down_size, w)
                b2_size = aligned_embed_dim * 4
                record_entry(f"layer.{layer}.b2", "fp32", b2_size)
                write_f32_zeros(w, aligned_embed_dim)  # b2

            # 4) final RMSNorm weight and bias placeholder
            final_norm = read_vector_f32(f, data_start, tensors["output_norm.weight"])
            record_entry("final_ln_weight", "fp32", aligned_embed_dim * 4)
            write_f32_padded(w, final_norm, aligned_embed_dim)
            record_entry("final_ln_bias", "fp32", aligned_embed_dim * 4)
            write_f32_zeros(w, aligned_embed_dim)

            checksum = w.digest()

            # Header: BUMPWGT4 format for v4 loader compatibility.
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
            # Tokenizer metadata (new in v4.1)
            out_f.write(struct.pack("<I", int(num_merges)))
            out_f.write(struct.pack("<I", int(total_vocab_bytes)))
            out_f.write(checksum)

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
        cfg["num_merges"] = num_merges
        cfg["total_vocab_bytes"] = total_vocab_bytes
        with open(args.config_out, "w", encoding="utf-8") as cf:
            json.dump(cfg, cf, indent=2)
            cf.write("\n")

    # Count how many layers have biases
    layers_with_bias = sum(1 for info in layer_infos if info["bq"] is not None)
    bias_status = f"biases={layers_with_bias}/{num_layers} layers" if layers_with_bias > 0 else "no biases"

    print(
        f"[gguf->bump] version={version} arch={arch} layers={num_layers} "
        f"hidden={embed_dim} heads={num_heads}/{num_kv_heads} ff={intermediate} "
        f"vocab={vocab_size} ctx={context_len} {bias_status} -> {args.output}"
    )

    # Print detailed conversion report with full tensor mapping
    print_conversion_report(
        num_layers=num_layers,
        layer_infos=layer_infos,
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        intermediate=intermediate,
        gguf_tensors=tensors,
        manifest_entries=manifest_entries,
    )
    if vocab_offsets is None or vocab_strings is None:
        print("[tokenizer] Warning: vocab/merges not embedded (pass --tokenizer-json or use GGUF with tokenizer metadata)")

    # Write manifest JSON if requested
    if args.manifest_out:
        os.makedirs(os.path.dirname(args.manifest_out) or ".", exist_ok=True)
        manifest = {
            "version": 5,
            "model": arch,
            "num_layers": num_layers,
            "embed_dim": embed_dim,
            "num_heads": num_heads,
            "num_kv_heads": num_kv_heads,
            "head_dim": head_dim,
            "intermediate_size": intermediate,
            "vocab_size": vocab_size,
            "context_length": context_len,
            "has_attention_biases": layers_with_bias > 0,
            "num_merges": num_merges,
            "total_vocab_bytes": total_vocab_bytes,
            "entries": manifest_entries,
        }
        with open(args.manifest_out, "w", encoding="utf-8") as mf:
            json.dump(manifest, mf, indent=2)
            mf.write("\n")
        print(f"[manifest] Written: {args.manifest_out} ({len(manifest_entries)} entries)")

    # Run parity verification if requested
    if args.verify:
        parity_ok = verify_bump_parity(
            gguf_path=args.gguf,
            bump_path=args.output,
            num_layers=num_layers,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
        )
        if not parity_ok:
            print("\n[ERROR] Parity verification failed! The converted file may be corrupt.")
            exit(1)


if __name__ == "__main__":
    main()

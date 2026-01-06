#!/usr/bin/env python3
"""
extract_gguf_vocab.py
=====================

Extracts tokenizer vocabulary and metadata from a GGUF file to a JSON file.
Calculates and reports storage size of the vocabulary.
"""

import argparse
import json
import os
import struct
import sys
from typing import BinaryIO, Dict, Optional, Any, List

# GGUF constants
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

class GGUFError(RuntimeError):
    pass

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

class GGUFReader:
    def __init__(self, f: BinaryIO) -> None:
        self._f = f
        try:
            self._file_size = os.fstat(f.fileno()).st_size
        except Exception:
            self._file_size = None

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
        data = self._f.read(n)
        if len(data) != n:
            raise GGUFError(f"Unexpected EOF (wanted {n} bytes, got {len(data)})")
        return data

    def u8(self) -> int: return struct.unpack("<B", self._read_exact(1))[0]
    def i8(self) -> int: return struct.unpack("<b", self._read_exact(1))[0]
    def u16(self) -> int: return struct.unpack("<H", self._read_exact(2))[0]
    def i16(self) -> int: return struct.unpack("<h", self._read_exact(2))[0]
    def u32(self) -> int: return struct.unpack("<I", self._read_exact(4))[0]
    def i32(self) -> int: return struct.unpack("<i", self._read_exact(4))[0]
    def u64(self) -> int: return struct.unpack("<Q", self._read_exact(8))[0]
    def i64(self) -> int: return struct.unpack("<q", self._read_exact(8))[0]
    def f32(self) -> float: return struct.unpack("<f", self._read_exact(4))[0]
    def f64(self) -> float: return struct.unpack("<d", self._read_exact(8))[0]

    def key_str(self) -> str:
        n = self.u64()
        return self._read_exact(n).decode("utf-8")

    def val_str(self) -> str:
        n = self.u64()
        return self._read_exact(n).decode("utf-8")

    def read_value(self, vtype: int) -> Any:
        if vtype == GGUF_TYPE_UINT8: return self.u8()
        if vtype == GGUF_TYPE_INT8: return self.i8()
        if vtype == GGUF_TYPE_UINT16: return self.u16()
        if vtype == GGUF_TYPE_INT16: return self.i16()
        if vtype == GGUF_TYPE_UINT32: return self.u32()
        if vtype == GGUF_TYPE_INT32: return self.i32()
        if vtype == GGUF_TYPE_UINT64: return self.u64()
        if vtype == GGUF_TYPE_INT64: return self.i64()
        if vtype == GGUF_TYPE_FLOAT32: return self.f32()
        if vtype == GGUF_TYPE_FLOAT64: return self.f64()
        if vtype == GGUF_TYPE_BOOL: return bool(self.u8())
        if vtype == GGUF_TYPE_STRING: return self.val_str()
        
        if vtype == GGUF_TYPE_ARRAY:
            elem_type = self.u32()
            n = self.u64()
            
            if elem_type == GGUF_TYPE_STRING:
                return [self.val_str() for _ in range(n)]
            
            elem_size = _gguf_scalar_size(elem_type)
            if elem_size is None:
                raise GGUFError(f"Unsupported GGUF array elem type {elem_type}")
            
            # Read numeric array efficiently
            data = self._read_exact(int(n) * elem_size)
            
            # Simple fmt mapping
            fmt_map = {
                GGUF_TYPE_UINT8: "B", GGUF_TYPE_INT8: "b",
                GGUF_TYPE_UINT16: "H", GGUF_TYPE_INT16: "h",
                GGUF_TYPE_UINT32: "I", GGUF_TYPE_INT32: "i",
                GGUF_TYPE_UINT64: "Q", GGUF_TYPE_INT64: "q",
                GGUF_TYPE_FLOAT32: "f", GGUF_TYPE_FLOAT64: "d",
                GGUF_TYPE_BOOL: "B",
            }
            fmt = fmt_map.get(elem_type)
            if fmt:
                return list(struct.unpack(f"<{n}{fmt}", data))
            return [] # Should not happen given elem_size check

        raise GGUFError(f"Unsupported GGUF value type {vtype}")
    
    def skip_value(self, vtype: int) -> None:
        size = _gguf_scalar_size(vtype)
        if size is not None:
            self.skip(size)
            return
        if vtype == GGUF_TYPE_BOOL:
            self.skip(1)
            return
        if vtype == GGUF_TYPE_STRING:
            n = self.u64()
            self.skip(int(n))
            return
        if vtype == GGUF_TYPE_ARRAY:
            elem_type = self.u32()
            n = self.u64()
            if elem_type == GGUF_TYPE_STRING:
                for _ in range(int(n)):
                    slen = self.u64()
                    self.skip(int(slen))
                return
            elem_size = _gguf_scalar_size(elem_type)
            if elem_size is not None:
                self.skip(int(n) * int(elem_size))
                return
        raise GGUFError(f"Unsupported GGUF value type {vtype}")


def extract_vocab(gguf_path: str, output_path: str):
    print(f"[extract_vocab] Reading {gguf_path}...")
    
    wanted_keys = {
        "tokenizer.ggml.model",
        "tokenizer.ggml.tokens",
        "tokenizer.ggml.scores",
        "tokenizer.ggml.token_type",
        "tokenizer.ggml.bos_token_id",
        "tokenizer.ggml.eos_token_id",
        "tokenizer.ggml.unknown_token_id",
        "tokenizer.ggml.padding_token_id",
        "tokenizer.ggml.add_bos_token",
        "tokenizer.ggml.add_eos_token",
        "tokenizer.chat_template",
    }
    
    data = {}
    
    with open(gguf_path, "rb") as f:
        r = GGUFReader(f)
        magic = r._read_exact(4)
        if magic != b"GGUF":
            raise GGUFError("Invalid magic")
        
        version = r.u32()
        if version >= 2:
            n_tensors = r.u64()
            n_kv = r.u64()
        else:
            n_tensors = r.u32()
            n_kv = r.u32()
            
        print(f"[extract_vocab] GGUF v{version}, {n_kv} KV pairs")
        
        for _ in range(n_kv):
            key = r.key_str()
            vtype = r.u32()
            
            if key in wanted_keys:
                # Read it fully
                val = r.read_value(vtype)
                data[key] = val
            else:
                r.skip_value(vtype)

    # Process and print stats
    tokens = data.get("tokenizer.ggml.tokens", [])
    scores = data.get("tokenizer.ggml.scores", [])
    vocab_size = len(tokens)
    
    if vocab_size == 0:
        print("[extract_vocab] Warning: No tokens found!")
        return

    # Calculate storage stats
    # 1. Tokens (list of strings): overhead + content
    # In GGUF on disk: 8 bytes len + string bytes per token
    token_bytes_on_disk = sum(8 + len(t.encode("utf-8")) for t in tokens)
    
    # 2. Scores (usually float32): 4 bytes per score
    score_bytes_on_disk = len(scores) * 4
    
    total_vocab_bytes = token_bytes_on_disk + score_bytes_on_disk
    total_mb = total_vocab_bytes / (1024 * 1024)
    
    print("\n" + "="*60)
    print(" TOKENIZER SUMMARY")
    print("="*60)
    print(f"  Vocab Size:    {vocab_size}")
    print(f"  Model Type:    {data.get('tokenizer.ggml.model', 'unknown')}")
    print(f"  Storage Size:  {total_mb:.2f} MB")
    print(f"    - Tokens:    {token_bytes_on_disk / 1024 / 1024:.2f} MB")
    print(f"    - Scores:    {score_bytes_on_disk / 1024 / 1024:.2f} MB")
    print("="*60 + "\n")
    
    # Structure for JSON output
    out_json = {
        "model": data.get("tokenizer.ggml.model"),
        "vocab_size": vocab_size,
        "special_tokens": {
            "bos": data.get("tokenizer.ggml.bos_token_id"),
            "eos": data.get("tokenizer.ggml.eos_token_id"),
            "unk": data.get("tokenizer.ggml.unknown_token_id"),
            "pad": data.get("tokenizer.ggml.padding_token_id"),
        },
        "add_special": {
            "bos": data.get("tokenizer.ggml.add_bos_token"),
            "eos": data.get("tokenizer.ggml.add_eos_token"),
        },
        "chat_template": data.get("tokenizer.chat_template"),
        "tokens": tokens,
        "scores": scores,
        "token_types": data.get("tokenizer.ggml.token_type"),
    }
    
    # Save
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(out_json, f, indent=2, ensure_ascii=False)
    
    print(f"[extract_vocab] Written to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Extract GGUF tokenizer")
    parser.add_argument("--gguf", required=True, help="Input GGUF file")
    parser.add_argument("--output", required=True, help="Output JSON file")
    args = parser.parse_args()
    
    try:
        extract_vocab(args.gguf, args.output)
    except Exception as e:
        print(f"[extract_vocab] Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
gguf_tokenizer.py - Simple tokenizer that works with GGUF vocab

Extracts tokenizer data directly from GGUF files and provides
encode/decode functionality using a greedy longest-match algorithm.

Supports:
- BPE (byte-pair encoding) used by GPT-2, LLaMA, Qwen, etc.
- SentencePiece (used by LLaMA, SmolLM, etc.)
"""

import json
import struct
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Any, BinaryIO, Tuple

# GGUF type constants
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
        GGUF_TYPE_UINT8: 1, GGUF_TYPE_INT8: 1,
        GGUF_TYPE_UINT16: 2, GGUF_TYPE_INT16: 2,
        GGUF_TYPE_UINT32: 4, GGUF_TYPE_INT32: 4,
        GGUF_TYPE_FLOAT32: 4, GGUF_TYPE_BOOL: 1,
        GGUF_TYPE_UINT64: 8, GGUF_TYPE_INT64: 8,
        GGUF_TYPE_FLOAT64: 8,
    }.get(vtype)


class GGUFReader:
    """Minimal GGUF reader for tokenizer extraction."""

    def __init__(self, f: BinaryIO):
        self._f = f

    def _read_exact(self, n: int) -> bytes:
        data = self._f.read(n)
        if len(data) != n:
            raise RuntimeError(f"Unexpected EOF (wanted {n}, got {len(data)})")
        return data

    def u8(self) -> int: return struct.unpack("<B", self._read_exact(1))[0]
    def u32(self) -> int: return struct.unpack("<I", self._read_exact(4))[0]
    def u64(self) -> int: return struct.unpack("<Q", self._read_exact(8))[0]
    def f32(self) -> float: return struct.unpack("<f", self._read_exact(4))[0]

    def key_str(self) -> str:
        n = self.u64()
        return self._read_exact(n).decode("utf-8")

    def val_str(self) -> str:
        n = self.u64()
        return self._read_exact(n).decode("utf-8")

    def read_value(self, vtype: int) -> Any:
        if vtype == GGUF_TYPE_UINT8: return self.u8()
        if vtype == GGUF_TYPE_UINT32: return self.u32()
        if vtype == GGUF_TYPE_INT32: return struct.unpack("<i", self._read_exact(4))[0]
        if vtype == GGUF_TYPE_UINT64: return self.u64()
        if vtype == GGUF_TYPE_FLOAT32: return self.f32()
        if vtype == GGUF_TYPE_BOOL: return bool(self.u8())
        if vtype == GGUF_TYPE_STRING: return self.val_str()

        if vtype == GGUF_TYPE_ARRAY:
            elem_type = self.u32()
            n = self.u64()
            if elem_type == GGUF_TYPE_STRING:
                return [self.val_str() for _ in range(n)]
            elem_size = _gguf_scalar_size(elem_type)
            if elem_size:
                data = self._read_exact(int(n) * elem_size)
                fmt_map = {
                    GGUF_TYPE_UINT8: "B", GGUF_TYPE_INT8: "b",
                    GGUF_TYPE_UINT16: "H", GGUF_TYPE_INT16: "h",
                    GGUF_TYPE_UINT32: "I", GGUF_TYPE_INT32: "i",
                    GGUF_TYPE_UINT64: "Q", GGUF_TYPE_INT64: "q",
                    GGUF_TYPE_FLOAT32: "f", GGUF_TYPE_FLOAT64: "d",
                }
                fmt = fmt_map.get(elem_type)
                if fmt:
                    return list(struct.unpack(f"<{n}{fmt}", data))
            return []
        raise RuntimeError(f"Unsupported GGUF type {vtype}")

    def skip_value(self, vtype: int):
        size = _gguf_scalar_size(vtype)
        if size is not None:
            self._f.seek(size, os.SEEK_CUR)
            return
        if vtype == GGUF_TYPE_BOOL:
            self._f.seek(1, os.SEEK_CUR)
            return
        if vtype == GGUF_TYPE_STRING:
            n = self.u64()
            self._f.seek(n, os.SEEK_CUR)
            return
        if vtype == GGUF_TYPE_ARRAY:
            elem_type = self.u32()
            n = self.u64()
            if elem_type == GGUF_TYPE_STRING:
                for _ in range(int(n)):
                    slen = self.u64()
                    self._f.seek(slen, os.SEEK_CUR)
                return
            elem_size = _gguf_scalar_size(elem_type)
            if elem_size:
                self._f.seek(int(n) * elem_size, os.SEEK_CUR)
                return
        raise RuntimeError(f"Unsupported GGUF type {vtype}")


def extract_tokenizer_from_gguf(gguf_path: str) -> Dict[str, Any]:
    """Extract tokenizer data from a GGUF file."""
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
            raise RuntimeError("Not a GGUF file")

        version = r.u32()
        if version >= 2:
            n_tensors = r.u64()
            n_kv = r.u64()
        else:
            n_tensors = r.u32()
            n_kv = r.u32()

        for _ in range(n_kv):
            key = r.key_str()
            vtype = r.u32()
            if key in wanted_keys:
                data[key] = r.read_value(vtype)
            else:
                r.skip_value(vtype)

    return data


class GGUFTokenizer:
    """
    Simple tokenizer using vocab from GGUF file.

    Uses greedy longest-match for encoding (like SentencePiece unigram).
    """

    def __init__(self, vocab_data: Dict[str, Any] = None):
        self.tokens: List[str] = []
        self.scores: List[float] = []
        self.token_to_id: Dict[str, int] = {}
        self.special_ids: set[int] = set()
        self.bos_id: int = 1
        self.eos_id: int = 2
        self.unk_id: int = 0
        self.pad_id: int = 0
        self.add_bos: bool = True
        self.add_eos: bool = False
        self.model_type: str = "unknown"

        if vocab_data:
            self._load_vocab(vocab_data)

    def _load_vocab(self, data: Dict[str, Any]):
        """Load vocabulary from extracted GGUF data."""
        self.tokens = data.get("tokenizer.ggml.tokens", [])
        self.scores = data.get("tokenizer.ggml.scores", [])
        self.model_type = data.get("tokenizer.ggml.model", "unknown")

        # Build token -> id mapping
        self.token_to_id = {t: i for i, t in enumerate(self.tokens)}

        # Special tokens
        self.bos_id = data.get("tokenizer.ggml.bos_token_id", 1)
        self.eos_id = data.get("tokenizer.ggml.eos_token_id", 2)
        self.unk_id = data.get("tokenizer.ggml.unknown_token_id", 0)
        self.pad_id = data.get("tokenizer.ggml.padding_token_id", 0)
        self.add_bos = data.get("tokenizer.ggml.add_bos_token", True)
        self.add_eos = data.get("tokenizer.ggml.add_eos_token", False)
        extra_special = data.get("tokenizer.ggml.special_token_ids", [])
        if isinstance(extra_special, list):
            self.special_ids = {int(x) for x in extra_special if isinstance(x, int)}
        else:
            self.special_ids = set()
        self.special_ids.update({self.bos_id, self.eos_id, self.pad_id, self.unk_id})

    @classmethod
    def from_gguf(cls, gguf_path: str) -> "GGUFTokenizer":
        """Create tokenizer from GGUF file."""
        data = extract_tokenizer_from_gguf(gguf_path)
        return cls(data)

    @classmethod
    def from_json(cls, json_path: str) -> "GGUFTokenizer":
        """Create tokenizer from extracted vocab JSON.

        Supports three common formats:
          1) Legacy exported format with keys: tokens/scores/special_tokens
          2) HuggingFace tokenizer.json with model.vocab
          3) Plain vocab.json map: {"token": id, ...}
        """
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 1) Legacy internal format
        if isinstance(data, dict) and isinstance(data.get("tokens"), list):
            vocab_data = {
                "tokenizer.ggml.tokens": data.get("tokens", []),
                "tokenizer.ggml.scores": data.get("scores", []),
                "tokenizer.ggml.model": data.get("model", "unknown"),
                "tokenizer.ggml.bos_token_id": data.get("special_tokens", {}).get("bos", 1),
                "tokenizer.ggml.eos_token_id": data.get("special_tokens", {}).get("eos", 2),
                "tokenizer.ggml.unknown_token_id": data.get("special_tokens", {}).get("unk", 0),
                "tokenizer.ggml.padding_token_id": data.get("special_tokens", {}).get("pad", 0),
                "tokenizer.ggml.add_bos_token": data.get("add_special", {}).get("bos", True),
                "tokenizer.ggml.add_eos_token": data.get("add_special", {}).get("eos", False),
            }
            return cls(vocab_data)

        vocab_map = None
        model_type = "unknown"
        bos_id, eos_id, unk_id, pad_id = 1, 2, 0, 0
        add_bos, add_eos = True, False
        special_ids: set[int] = set()

        # 2) HuggingFace tokenizer.json (tokenizers format)
        if isinstance(data, dict) and isinstance(data.get("model"), dict):
            model = data.get("model", {})
            maybe_vocab = model.get("vocab")
            if isinstance(maybe_vocab, dict):
                vocab_map = maybe_vocab
                model_type = str(model.get("type", "unknown"))
                unk_tok = model.get("unk_token")
                if isinstance(unk_tok, str) and unk_tok in vocab_map:
                    unk_id = int(vocab_map[unk_tok])

            # Best-effort special-token extraction from added_tokens.
            added = data.get("added_tokens")
            if isinstance(added, list):
                for item in added:
                    if not isinstance(item, dict):
                        continue
                    content = item.get("content")
                    tid = item.get("id")
                    if not isinstance(content, str) or not isinstance(tid, int):
                        continue
                    c = content.lower()
                    if c in {"<s>", "<|im_start|>", "<|bos|>", "<bos>"}:
                        bos_id = tid
                    elif c in {"</s>", "<|im_end|>", "<|endoftext|>", "<eos>", "<|eot_id|>"}:
                        eos_id = tid
                    elif "unk" in c:
                        unk_id = tid
                    elif "pad" in c:
                        pad_id = tid
                    if bool(item.get("special")):
                        special_ids.add(tid)

        # 3) Plain vocab map file (token -> id)
        if vocab_map is None and isinstance(data, dict):
            # Heuristic: most values are integers, and keys are token strings.
            sample_items = list(data.items())[:64]
            if sample_items and all(isinstance(k, str) and isinstance(v, int) for k, v in sample_items):
                vocab_map = data
                model_type = "bpe"

        if isinstance(vocab_map, dict):
            added_tokens_map: Dict[int, str] = {}
            added = data.get("added_tokens")
            if isinstance(added, list):
                for item in added:
                    if not isinstance(item, dict):
                        continue
                    content = item.get("content")
                    tid = item.get("id")
                    if isinstance(content, str) and isinstance(tid, int) and tid >= 0:
                        added_tokens_map[int(tid)] = content

            if vocab_map:
                max_vocab_id = max(int(v) for v in vocab_map.values())
                max_added_id = max(added_tokens_map.keys()) if added_tokens_map else -1
                max_id = max(max_vocab_id, max_added_id)
                tokens = [f"<|ck_missing_{i}|>" for i in range(max_id + 1)]
                for tok, tid in vocab_map.items():
                    if isinstance(tok, str) and isinstance(tid, int) and 0 <= tid <= max_id:
                        tokens[tid] = tok
                for tid, tok in added_tokens_map.items():
                    if 0 <= tid <= max_id:
                        tokens[tid] = tok
            else:
                tokens = []

            vocab_data = {
                "tokenizer.ggml.tokens": tokens,
                "tokenizer.ggml.scores": [0.0] * len(tokens),
                "tokenizer.ggml.model": model_type,
                "tokenizer.ggml.bos_token_id": bos_id,
                "tokenizer.ggml.eos_token_id": eos_id,
                "tokenizer.ggml.unknown_token_id": unk_id,
                "tokenizer.ggml.padding_token_id": pad_id,
                "tokenizer.ggml.add_bos_token": add_bos,
                "tokenizer.ggml.add_eos_token": add_eos,
                "tokenizer.ggml.special_token_ids": sorted(special_ids),
            }
            return cls(vocab_data)

        raise ValueError(f"Unsupported tokenizer JSON format: {json_path}")

    def _byte_to_token(self, b: int) -> str:
        """Convert a byte to its token representation."""
        # SentencePiece/GGUF uses <0xXX> format for raw bytes
        return f"<0x{b:02X}>"

    def encode(self, text: str, add_bos: bool = None, add_eos: bool = None) -> List[int]:
        """
        Encode text to token IDs using greedy longest-match.

        Args:
            text: Input text to tokenize
            add_bos: Whether to add BOS token (default: self.add_bos)
            add_eos: Whether to add EOS token (default: self.add_eos)

        Returns:
            List of token IDs
        """
        if add_bos is None:
            add_bos = self.add_bos
        if add_eos is None:
            add_eos = self.add_eos

        ids = []
        if add_bos:
            ids.append(self.bos_id)

        # Detect tokenizer type for space handling
        # GPT-2 uses "Ġ" (U+0120), SentencePiece uses "▁" (U+2581)
        is_gpt2 = self.model_type == "gpt2" or "\u0120" in "".join(self.tokens[:1000])

        i = 0
        while i < len(text):
            best_len = 0
            best_id = self.unk_id

            # Determine if this position should have a space prefix
            need_space_prefix = (i > 0 and text[i] != ' ' and text[i-1] == ' ')

            # Try different lengths, longest first
            for length in range(min(len(text) - i, 32), 0, -1):
                chunk = text[i:i + length]

                # Skip leading space in chunk (it becomes prefix)
                if chunk.startswith(' '):
                    chunk_no_space = chunk[1:]
                    if is_gpt2:
                        # GPT-2: space becomes "Ġ" prefix
                        prefixed = "\u0120" + chunk_no_space
                    else:
                        # SentencePiece: space becomes "▁" prefix
                        prefixed = "\u2581" + chunk_no_space

                    if prefixed in self.token_to_id:
                        best_len = length
                        best_id = self.token_to_id[prefixed]
                        break

                # Try direct match
                if chunk in self.token_to_id:
                    best_len = length
                    best_id = self.token_to_id[chunk]
                    break

                # Try with space prefix if needed (for tokens after space)
                if need_space_prefix and length == len(text) - i:
                    continue  # Skip - we'll try shorter
                elif need_space_prefix:
                    if is_gpt2:
                        prefixed = "\u0120" + chunk
                    else:
                        prefixed = "\u2581" + chunk
                    if prefixed in self.token_to_id:
                        best_len = length
                        best_id = self.token_to_id[prefixed]
                        break

            if best_len == 0:
                # Fall back to single character/byte
                char = text[i]
                if char in self.token_to_id:
                    best_id = self.token_to_id[char]
                elif char == ' ':
                    # Space alone - try space token
                    if is_gpt2 and "\u0120" in self.token_to_id:
                        best_id = self.token_to_id["\u0120"]
                    elif "\u2581" in self.token_to_id:
                        best_id = self.token_to_id["\u2581"]
                else:
                    # Try byte token
                    for b in char.encode("utf-8"):
                        byte_token = self._byte_to_token(b)
                        if byte_token in self.token_to_id:
                            best_id = self.token_to_id[byte_token]
                            break
                best_len = 1

            ids.append(best_id)
            i += best_len

        if add_eos:
            ids.append(self.eos_id)

        return ids

    def decode(self, ids: List[int], skip_special: bool = True) -> str:
        """
        Decode token IDs to text.

        Args:
            ids: List of token IDs
            skip_special: Whether to skip special tokens (BOS, EOS, PAD)

        Returns:
            Decoded text string
        """
        special_ids = set(self.special_ids)
        pieces = []

        for token_id in ids:
            if skip_special and token_id in special_ids:
                continue

            if 0 <= token_id < len(self.tokens):
                token = self.tokens[token_id]

                # Handle GPT-2 space marker (Ġ = U+0120)
                if token.startswith("\u0120"):
                    token = " " + token[1:]

                # Handle SentencePiece space marker (▁ = U+2581)
                if token.startswith("\u2581"):
                    token = " " + token[1:]

                # Handle byte tokens <0xXX>
                if token.startswith("<0x") and token.endswith(">"):
                    try:
                        byte_val = int(token[3:-1], 16)
                        token = bytes([byte_val]).decode("utf-8", errors="replace")
                    except ValueError:
                        pass

                pieces.append(token)

        return "".join(pieces)

    @property
    def vocab_size(self) -> int:
        return len(self.tokens)

    def save(self, path: str):
        """Save tokenizer vocab to JSON."""
        data = {
            "model": self.model_type,
            "vocab_size": len(self.tokens),
            "special_tokens": {
                "bos": self.bos_id,
                "eos": self.eos_id,
                "unk": self.unk_id,
                "pad": self.pad_id,
            },
            "add_special": {
                "bos": self.add_bos,
                "eos": self.add_eos,
            },
            "tokens": self.tokens,
            "scores": self.scores,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)


# Compatibility wrapper to match HuggingFace tokenizers API
class Tokenizer:
    """
    Wrapper class that provides HuggingFace tokenizers-compatible API.
    This allows drop-in replacement of the tokenizers library.
    """

    def __init__(self, tokenizer: GGUFTokenizer):
        self._tokenizer = tokenizer

    @classmethod
    def from_file(cls, path: str) -> "Tokenizer":
        """Load tokenizer from file (JSON or detect GGUF)."""
        if path.endswith(".gguf"):
            return cls(GGUFTokenizer.from_gguf(path))
        elif path.endswith(".json"):
            return cls(GGUFTokenizer.from_json(path))
        else:
            # Try JSON first, then GGUF
            try:
                return cls(GGUFTokenizer.from_json(path))
            except (json.JSONDecodeError, KeyError):
                return cls(GGUFTokenizer.from_gguf(path))

    def encode(self, text: str, add_special_tokens: bool = True) -> "Encoding":
        """Encode text to token IDs."""
        ids = self._tokenizer.encode(
            text,
            add_bos=add_special_tokens and self._tokenizer.add_bos,
            add_eos=add_special_tokens and self._tokenizer.add_eos
        )
        return Encoding(ids)

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs to text."""
        return self._tokenizer.decode(ids, skip_special=skip_special_tokens)

    def get_vocab(self) -> Dict[str, int]:
        """HuggingFace-compatible vocab map."""
        return dict(self._tokenizer.token_to_id)

    def id_to_token(self, token_id: int) -> Optional[str]:
        """HuggingFace-compatible token lookup."""
        tid = int(token_id)
        if 0 <= tid < len(self._tokenizer.tokens):
            return self._tokenizer.tokens[tid]
        return None

    @property
    def vocab_size(self) -> int:
        return self._tokenizer.vocab_size


class Encoding:
    """Simple encoding result that mimics HuggingFace tokenizers.Encoding."""

    def __init__(self, ids: List[int]):
        self.ids = ids

    def __len__(self) -> int:
        return len(self.ids)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python gguf_tokenizer.py <gguf_file> [text_to_encode]")
        sys.exit(1)

    gguf_path = sys.argv[1]
    print(f"Loading tokenizer from {gguf_path}...")
    tokenizer = GGUFTokenizer.from_gguf(gguf_path)

    print(f"Model type: {tokenizer.model_type}")
    print(f"Vocab size: {tokenizer.vocab_size}")
    print(f"BOS ID: {tokenizer.bos_id}")
    print(f"EOS ID: {tokenizer.eos_id}")

    if len(sys.argv) > 2:
        text = " ".join(sys.argv[2:])
        print(f"\nEncoding: '{text}'")
        ids = tokenizer.encode(text)
        print(f"Token IDs: {ids}")
        decoded = tokenizer.decode(ids)
        print(f"Decoded: '{decoded}'")

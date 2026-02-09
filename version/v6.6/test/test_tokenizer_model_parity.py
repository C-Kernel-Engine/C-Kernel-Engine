#!/usr/bin/env python3
"""
Regression test: C tokenizer parity vs GGUF reference tokenizer.

Uses the lightweight tokenizer runtime (`libckernel_tokenizer.so`) instead of
full `libmodel.so` initialization, so this regression still runs in low-memory
environments while catching tokenizer behavior drift.
"""

from __future__ import annotations

import argparse
import ast
import ctypes
import json
import os
import shutil
import struct
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
RESET = "\033[0m"

# GGUF value types
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

# tokenizer.h enums
CK_TOKENIZER_SPM = 2
CK_SPM_MODE_UNIGRAM = 0
CK_SPM_MODE_LLAMA = 1


def _log(prefix: str, msg: str) -> None:
    print(f"{prefix} {msg}{RESET}")


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


class _GGUFReader:
    def __init__(self, f):
        self._f = f

    def _read_exact(self, n: int) -> bytes:
        data = self._f.read(n)
        if len(data) != n:
            raise RuntimeError(f"Unexpected EOF (wanted {n}, got {len(data)})")
        return data

    def u8(self) -> int:
        return struct.unpack("<B", self._read_exact(1))[0]

    def u32(self) -> int:
        return struct.unpack("<I", self._read_exact(4))[0]

    def i32(self) -> int:
        return struct.unpack("<i", self._read_exact(4))[0]

    def u64(self) -> int:
        return struct.unpack("<Q", self._read_exact(8))[0]

    def f32(self) -> float:
        return struct.unpack("<f", self._read_exact(4))[0]

    def val_str(self) -> str:
        n = self.u64()
        return self._read_exact(n).decode("utf-8")

    def key_str(self) -> str:
        return self.val_str()

    def read_value(self, vtype: int) -> Any:
        if vtype == GGUF_TYPE_UINT8:
            return self.u8()
        if vtype == GGUF_TYPE_INT32:
            return self.i32()
        if vtype == GGUF_TYPE_UINT32:
            return self.u32()
        if vtype == GGUF_TYPE_UINT64:
            return self.u64()
        if vtype == GGUF_TYPE_FLOAT32:
            return self.f32()
        if vtype == GGUF_TYPE_BOOL:
            return bool(self.u8())
        if vtype == GGUF_TYPE_STRING:
            return self.val_str()
        if vtype == GGUF_TYPE_ARRAY:
            elem_type = self.u32()
            n = self.u64()
            if elem_type == GGUF_TYPE_STRING:
                return [self.val_str() for _ in range(n)]
            elem_size = _gguf_scalar_size(elem_type)
            if elem_size is None:
                raise RuntimeError(f"Unsupported GGUF array elem type {elem_type}")
            data = self._read_exact(int(n) * elem_size)
            fmt_map = {
                GGUF_TYPE_UINT8: "B",
                GGUF_TYPE_INT8: "b",
                GGUF_TYPE_UINT16: "H",
                GGUF_TYPE_INT16: "h",
                GGUF_TYPE_UINT32: "I",
                GGUF_TYPE_INT32: "i",
                GGUF_TYPE_UINT64: "Q",
                GGUF_TYPE_INT64: "q",
                GGUF_TYPE_FLOAT32: "f",
                GGUF_TYPE_FLOAT64: "d",
            }
            fmt = fmt_map.get(elem_type)
            if fmt is None:
                return []
            return list(struct.unpack(f"<{n}{fmt}", data))
        raise RuntimeError(f"Unsupported GGUF type {vtype}")

    def skip_value(self, vtype: int) -> None:
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
            if elem_size is None:
                raise RuntimeError(f"Unsupported GGUF array elem type {elem_type}")
            self._f.seek(int(n) * elem_size, os.SEEK_CUR)
            return
        raise RuntimeError(f"Unsupported GGUF type {vtype}")


def _extract_gguf_tokenizer(gguf_path: Path) -> Dict[str, Any]:
    wanted = {
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
        "tokenizer.ggml.add_space_prefix",
    }
    meta: Dict[str, Any] = {}

    with open(gguf_path, "rb") as f:
        r = _GGUFReader(f)
        magic = r._read_exact(4)
        if magic != b"GGUF":
            raise RuntimeError(f"{gguf_path} is not a GGUF file")
        version = r.u32()
        if version >= 2:
            _n_tensors = r.u64()
            n_kv = r.u64()
        else:
            _n_tensors = r.u32()
            n_kv = r.u32()

        for _ in range(n_kv):
            key = r.key_str()
            vtype = r.u32()
            if key in wanted:
                meta[key] = r.read_value(vtype)
            else:
                r.skip_value(vtype)

    tokens = meta.get("tokenizer.ggml.tokens")
    if not isinstance(tokens, list) or not tokens:
        raise RuntimeError("GGUF tokenizer.ggml.tokens is missing/empty")

    vocab_size = len(tokens)
    scores = meta.get("tokenizer.ggml.scores")
    if not isinstance(scores, list):
        scores = []
    types = meta.get("tokenizer.ggml.token_type")
    if not isinstance(types, list):
        types = []

    while len(scores) < vocab_size:
        scores.append(0.0)
    while len(types) < vocab_size:
        types.append(1)

    return {
        "tokens": tokens,
        "scores": [float(x) for x in scores[:vocab_size]],
        "types": [int(x) for x in types[:vocab_size]],
        "model": meta.get("tokenizer.ggml.model"),
        "bos_id": meta.get("tokenizer.ggml.bos_token_id"),
        "eos_id": meta.get("tokenizer.ggml.eos_token_id"),
        "unk_id": meta.get("tokenizer.ggml.unknown_token_id"),
        "pad_id": meta.get("tokenizer.ggml.padding_token_id"),
        "add_bos": meta.get("tokenizer.ggml.add_bos_token"),
        "add_eos": meta.get("tokenizer.ggml.add_eos_token"),
        "add_space_prefix": meta.get("tokenizer.ggml.add_space_prefix"),
    }


def _find_repo_root(start: Path) -> Path:
    curr = start
    for parent in [curr] + list(curr.parents):
        if (parent / ".git").exists() and (parent / "scripts").exists():
            return parent
    return Path(__file__).resolve().parents[3]


def _find_default_model_pair() -> Tuple[Optional[Path], Optional[Path]]:
    roots = [
        Path.home() / ".cache" / "ck-engine-v6.6" / "models",
        Path.home() / ".cache" / "ck-engine-v6" / "models",
    ]
    candidates = []
    for root in roots:
        if not root.exists():
            continue
        for base in root.iterdir():
            if not base.is_dir():
                continue
            ck_dir = base / "ck_build"
            if not (ck_dir / "init_call.json").exists():
                continue
            ggufs = sorted(base.glob("*.gguf"))
            if not ggufs:
                continue
            mtime = (ck_dir / "init_call.json").stat().st_mtime
            candidates.append((mtime, ck_dir, ggufs[0]))
    if not candidates:
        return None, None
    _, ck_dir, gguf_path = sorted(candidates, key=lambda x: x[0], reverse=True)[0]
    return ck_dir, gguf_path


def _load_init_special_tokens(model_dir: Path) -> Dict[str, Any]:
    init_call = model_dir / "init_call.json"
    if not init_call.exists():
        return {}
    try:
        with open(init_call, "r", encoding="utf-8") as f:
            obj = json.load(f)
    except Exception:
        return {}
    special = obj.get("special_tokens", {}) or {}
    return special if isinstance(special, dict) else {}


class CKTokenizerRuntime:
    def __init__(self, repo_root: Path, model_dir: Path, gguf_path: Path):
        self.repo_root = repo_root
        self.model_dir = model_dir
        self.gguf_path = gguf_path
        self.lib = None
        self.tok = None

    def load(self) -> None:
        lib_path = self.repo_root / "build" / "libckernel_tokenizer.so"
        if not lib_path.exists():
            lib_path = self.model_dir / "libckernel_tokenizer.so"
        if not lib_path.exists():
            raise RuntimeError("Could not find libckernel_tokenizer.so")

        self.lib = ctypes.CDLL(str(lib_path))
        self.lib.ck_tokenizer_create.restype = ctypes.c_void_p
        self.lib.ck_tokenizer_create.argtypes = [ctypes.c_int]
        self.lib.ck_tokenizer_free.argtypes = [ctypes.c_void_p]
        self.lib.ck_tokenizer_load_binary_with_scores.restype = ctypes.c_int
        self.lib.ck_tokenizer_load_binary_with_scores.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int32),
            ctypes.c_char_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        self.lib.ck_tokenizer_set_special_ids.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_int32,
        ]
        self.lib.ck_tokenizer_set_add_bos_eos.argtypes = [
            ctypes.c_void_p,
            ctypes.c_bool,
            ctypes.c_bool,
        ]
        self.lib.ck_tokenizer_set_add_space_prefix.argtypes = [
            ctypes.c_void_p,
            ctypes.c_bool,
        ]
        self.lib.ck_tokenizer_set_spm_mode.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
        ]
        self.lib.ck_tokenizer_encode.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int32),
            ctypes.c_int,
        ]
        self.lib.ck_tokenizer_encode.restype = ctypes.c_int

        gguf = _extract_gguf_tokenizer(self.gguf_path)
        tokens: List[str] = gguf["tokens"]
        scores: List[float] = gguf["scores"]
        types: List[int] = gguf["types"]
        vocab_size = len(tokens)

        offsets: List[int] = []
        pool = bytearray()
        for token in tokens:
            offsets.append(len(pool))
            pool.extend(token.encode("utf-8", errors="replace"))
            pool.append(0)

        off_arr = (ctypes.c_int32 * vocab_size)(*offsets)
        score_arr = (ctypes.c_float * vocab_size)(*scores)
        type_arr = (ctypes.c_uint8 * vocab_size)(*types)
        strings_blob = bytes(pool)

        self.tok = self.lib.ck_tokenizer_create(CK_TOKENIZER_SPM)
        if not self.tok:
            raise RuntimeError("ck_tokenizer_create failed")

        rc = self.lib.ck_tokenizer_load_binary_with_scores(
            self.tok,
            vocab_size,
            off_arr,
            strings_blob,
            score_arr,
            type_arr,
            0,
            None,
        )
        if rc != 0:
            raise RuntimeError(f"ck_tokenizer_load_binary_with_scores failed: {rc}")

        special = _load_init_special_tokens(self.model_dir)

        def _pick(key: str, fallback_key: str, default: Any) -> Any:
            if key in special:
                return special[key]
            if fallback_key in gguf and gguf[fallback_key] is not None:
                return gguf[fallback_key]
            return default

        unk_id = int(_pick("unk_token_id", "unk_id", 0))
        bos_id = int(_pick("bos_token_id", "bos_id", -1))
        eos_id = int(_pick("eos_token_id", "eos_id", -1))
        pad_id = int(_pick("pad_token_id", "pad_id", -1))

        add_bos = bool(_pick("add_bos_token", "add_bos", True))
        add_eos = bool(_pick("add_eos_token", "add_eos", False))
        add_space_prefix = bool(_pick("add_space_prefix", "add_space_prefix", True))

        tokenizer_model = special.get("tokenizer_model", gguf.get("model"))
        spm_mode = CK_SPM_MODE_LLAMA if str(tokenizer_model).strip().lower() == "llama" else CK_SPM_MODE_UNIGRAM

        self.lib.ck_tokenizer_set_special_ids(self.tok, unk_id, bos_id, eos_id, pad_id, -1)
        self.lib.ck_tokenizer_set_add_bos_eos(self.tok, add_bos, add_eos)
        self.lib.ck_tokenizer_set_add_space_prefix(self.tok, add_space_prefix)
        self.lib.ck_tokenizer_set_spm_mode(self.tok, spm_mode)

    def encode(self, text: str) -> List[int]:
        if self.lib is None or self.tok is None:
            raise RuntimeError("Tokenizer is not loaded")
        text_bytes = text.encode("utf-8")
        out = (ctypes.c_int32 * 4096)()
        n = self.lib.ck_tokenizer_encode(self.tok, text_bytes, len(text_bytes), out, 4096)
        if n <= 0:
            return []
        return [out[i] for i in range(n)]

    def close(self) -> None:
        if self.lib is not None and self.tok:
            self.lib.ck_tokenizer_free(self.tok)
            self.tok = None


def _llama_tokenize_ids(llama_tokenize_bin: Path, gguf_path: Path, text: str) -> List[int]:
    cmd = [str(llama_tokenize_bin), "-m", str(gguf_path), "--ids", "--log-disable", "-p", text]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"llama-tokenize failed (rc={result.returncode}): {result.stderr.strip()}")
    out = result.stdout.strip()
    try:
        ids = ast.literal_eval(out)
    except Exception as exc:
        raise RuntimeError(f"Unexpected llama-tokenize output: {out}") from exc
    if not isinstance(ids, list):
        raise RuntimeError(f"Unexpected llama-tokenize output type: {type(ids)}")
    return [int(x) for x in ids]


def _fallback_gguf_ids(gguf_path: Path, text: str, repo_root: Path) -> List[int]:
    # Fallback only if llama-tokenize binary is unavailable.
    sys.path.insert(0, str(repo_root / "scripts"))
    from gguf_tokenizer import GGUFTokenizer, Tokenizer as GGUFTokenizerWrapper  # type: ignore

    tokenizer = GGUFTokenizerWrapper(GGUFTokenizer.from_gguf(str(gguf_path)))
    return list(tokenizer.encode(text).ids)


def _reference_encode(
    gguf_path: Path, text: str, llama_tokenize_bin: Optional[Path], repo_root: Path
) -> Tuple[List[int], str]:
    if llama_tokenize_bin and llama_tokenize_bin.exists():
        return _llama_tokenize_ids(llama_tokenize_bin, gguf_path, text), "llama-tokenize"
    return _fallback_gguf_ids(gguf_path, text, repo_root), "gguf_tokenizer_fallback"


def _probe_texts() -> Iterable[str]:
    return [
        "hello",
        "Hello world",
        "  leading space",
        "multiple   spaces",
        "special chars @#$%",
        "cafe",
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Tokenizer parity regression test")
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=os.environ.get("CK_TOKENIZER_MODEL_DIR", None),
        help="Path to ck_build directory containing init_call.json",
    )
    parser.add_argument(
        "--gguf",
        type=Path,
        default=os.environ.get("CK_TOKENIZER_GGUF", None),
        help="Path to GGUF model used as tokenizer reference",
    )
    parser.add_argument(
        "--llama-tokenize-bin",
        type=Path,
        default=None,
        help="Path to llama.cpp llama-tokenize binary",
    )
    parser.add_argument(
        "--strict-missing",
        action="store_true",
        help="Fail if model/gguf paths cannot be resolved",
    )
    args = parser.parse_args()

    repo_root = _find_repo_root(Path(__file__).resolve())

    model_dir = args.model_dir
    gguf_path = args.gguf
    if model_dir is None or gguf_path is None:
        auto_model_dir, auto_gguf = _find_default_model_pair()
        model_dir = model_dir or auto_model_dir
        gguf_path = gguf_path or auto_gguf

    if model_dir is None or gguf_path is None:
        msg = (
            "No model pair found. Set --model-dir/--gguf (or env CK_TOKENIZER_MODEL_DIR/"
            "CK_TOKENIZER_GGUF) to run this regression."
        )
        if args.strict_missing:
            _log(RED + "[FAIL]", msg)
            return 1
        _log(YELLOW + "[SKIP]", msg)
        return 0

    if not model_dir.exists() or not gguf_path.exists():
        _log(RED + "[FAIL]", f"Invalid paths: model_dir={model_dir}, gguf={gguf_path}")
        return 1

    _log(CYAN + "[INFO]", f"Model dir: {model_dir}")
    _log(CYAN + "[INFO]", f"GGUF:      {gguf_path}")

    if args.llama_tokenize_bin is not None:
        llama_tokenize_bin = args.llama_tokenize_bin
    elif os.environ.get("LLAMA_TOKENIZE_BIN"):
        llama_tokenize_bin = Path(os.environ["LLAMA_TOKENIZE_BIN"])
    else:
        default_bin = repo_root / "llama.cpp" / "build" / "bin" / "llama-tokenize"
        if default_bin.exists():
            llama_tokenize_bin = default_bin
        else:
            system_bin = shutil.which("llama-tokenize")
            llama_tokenize_bin = Path(system_bin) if system_bin else None

    if llama_tokenize_bin and not llama_tokenize_bin.exists():
        _log(YELLOW + "[WARN]", f"{llama_tokenize_bin} not found; falling back to scripts/gguf_tokenizer.py")
        llama_tokenize_bin = None

    runtime = CKTokenizerRuntime(repo_root, model_dir, gguf_path)
    try:
        runtime.load()
    except Exception as exc:
        _log(RED + "[FAIL]", f"Could not initialize C tokenizer runtime: {exc}")
        return 1

    failures = 0
    try:
        for text in _probe_texts():
            try:
                ref_ids, source = _reference_encode(gguf_path, text, llama_tokenize_bin, repo_root)
            except Exception as exc:
                _log(RED + "[FAIL]", f"Reference encode failed for {text!r}: {exc}")
                failures += 1
                continue

            try:
                ck_ids = runtime.encode(text)
            except Exception as exc:
                _log(RED + "[FAIL]", f"C encode failed for {text!r}: {exc}")
                failures += 1
                continue

            if ck_ids != ref_ids:
                failures += 1
                _log(RED + "[FAIL]", f"text={text!r}")
                print(f"       C tokenizer ids:   {ck_ids}")
                print(f"       {source} ids: {ref_ids}")
            else:
                _log(GREEN + "[PASS]", f"text={text!r} ids={ck_ids}")
    finally:
        runtime.close()

    if failures:
        _log(RED + "[FAIL]", f"Tokenizer parity regression failed ({failures} mismatches)")
        return 1

    _log(GREEN + "[PASS]", "Tokenizer parity regression passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

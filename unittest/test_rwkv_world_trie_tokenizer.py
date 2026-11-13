#!/usr/bin/env python3
"""Parity checks for the native RWKV World byte-level trie tokenizer.

The C backend stores the WordPiece-shaped ByteLevel vocabulary verbatim and
encodes UTF-8 input with greedy longest-match. This test compares it against a
faithful Python transcription of the reference tokenizer's rules ("highest id
along the matched path"), and against the publisher's `Rwkv7Tokenizer` when
that class is available locally.

Usage:
    make tokenizer
    .venv/bin/python unittest/test_rwkv_world_trie_tokenizer.py

Set CK_RWKV_TOKENIZER_JSON to use a different tokenizer.json. The test skips
when no RWKV World vocabulary can be found.
"""

from __future__ import annotations

import ctypes
import importlib.util
import json
import os
import subprocess
import tempfile
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "version" / "v8" / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

LIB_PATH = ROOT / "build" / "libckernel_tokenizer.so"
CK_TOKENIZER_RWKV_WORLD_TRIE = 3


def _find_tokenizer_json() -> Path | None:
    override = os.environ.get("CK_RWKV_TOKENIZER_JSON")
    candidates = [Path(override)] if override else []
    candidates.append(ROOT / "unittest" / "tokenizer.json")
    for path in candidates:
        if path.is_file():
            return path
    return None


def _bytes_to_unicode() -> dict[int, str]:
    values = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    characters = list(values)
    extra = 0
    for byte in range(256):
        if byte not in values:
            values.append(byte)
            characters.append(256 + extra)
            extra += 1
    return dict(zip(values, map(chr, characters)))


class ReferenceTokenizer:
    """Faithful port of Rwkv7Tokenizer's byte-level longest-prefix rules."""

    def __init__(self, vocabulary: dict[str, int]) -> None:
        decoder = {
            character: byte for byte, character in _bytes_to_unicode().items()
        }
        self._terminal: dict[bytes, int] = {}
        self.raw_by_id: dict[int, bytes] = {}
        self._added: list[tuple[bytes, int]] = []
        for token, index in vocabulary.items():
            if index == 0:
                # The reference reserves id 0 as an added token matched before
                # ordinary encoding, and keeps it out of the byte trie.
                raw = token.encode("utf-8")
                self.raw_by_id[index] = raw
                self._added.append((raw, index))
                continue
            if token == f"\ue000{index}\ue001":
                self.raw_by_id[index] = token.encode("utf-8")
                continue
            raw = bytes(decoder[character] for character in token)
            self.raw_by_id[index] = raw
            self._terminal[raw] = index
        self._added.sort(key=lambda item: -len(item[0]))
        self.max_token_bytes = max(len(raw) for raw in self._terminal)

    def _encode_ordinary(self, data: bytes) -> list[int]:
        output: list[int] = []
        position = 0
        while position < len(data):
            best_id: int | None = None
            best_end = position
            limit = min(len(data), position + self.max_token_bytes)
            for end in range(position + 1, limit + 1):
                token_id = self._terminal.get(data[position:end])
                if token_id is not None and (best_id is None or token_id > best_id):
                    best_id = token_id
                    best_end = end
            if best_id is None:
                raise RuntimeError(
                    f"singleton-byte vocabulary invariant violated at {position}"
                )
            output.append(best_id)
            position = best_end
        return output

    def encode(self, data: bytes) -> list[int]:
        output: list[int] = []
        ordinary_start = 0
        position = 0
        while position < len(data):
            matched = next(
                (
                    item
                    for item in self._added
                    if data.startswith(item[0], position)
                ),
                None,
            )
            if matched is None:
                position += 1
                continue
            output.extend(self._encode_ordinary(data[ordinary_start:position]))
            output.append(matched[1])
            position += len(matched[0])
            ordinary_start = position
        output.extend(self._encode_ordinary(data[ordinary_start:]))
        return output

    def decode(self, ids: list[int]) -> bytes:
        return b"".join(self.raw_by_id[token_id] for token_id in ids)


def _load_reference_class(path: Path):
    try:
        import transformers  # noqa: F401
    except ImportError:
        return None
    spec = importlib.util.spec_from_file_location("rwkv7_tokenizer_reference", path)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return getattr(module, "Rwkv7Tokenizer", None)


def _load_library():
    if not LIB_PATH.is_file():
        raise SystemExit(
            f"tokenizer library not found: {LIB_PATH}\nRun 'make tokenizer' first."
        )
    lib = ctypes.CDLL(str(LIB_PATH))
    lib.ck_tokenizer_create.restype = ctypes.c_void_p
    lib.ck_tokenizer_create.argtypes = [ctypes.c_int]
    lib.ck_tokenizer_free.argtypes = [ctypes.c_void_p]
    lib.ck_tokenizer_set_use_trie.argtypes = [ctypes.c_void_p, ctypes.c_bool]
    lib.ck_tokenizer_set_add_bos_eos.argtypes = [
        ctypes.c_void_p, ctypes.c_bool, ctypes.c_bool,
    ]
    lib.ck_tokenizer_set_special_ids.argtypes = [
        ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
        ctypes.c_int32, ctypes.c_int32,
    ]
    lib.ck_tokenizer_load_binary.restype = ctypes.c_int
    lib.ck_tokenizer_load_binary.argtypes = [
        ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_int32),
        ctypes.c_char_p, ctypes.c_int, ctypes.POINTER(ctypes.c_int32),
    ]
    lib.ck_tokenizer_encode.restype = ctypes.c_int
    lib.ck_tokenizer_encode.argtypes = [
        ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int,
        ctypes.POINTER(ctypes.c_int32), ctypes.c_int,
    ]
    lib.ck_tokenizer_decode.restype = ctypes.c_int
    lib.ck_tokenizer_decode.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32), ctypes.c_int,
        ctypes.c_char_p, ctypes.c_int,
    ]
    lib.ck_tokenizer_lookup.restype = ctypes.c_int32
    lib.ck_tokenizer_lookup.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    return lib


def _prompt_cases() -> list[bytes]:
    cases = [
        b"Hello world",
        b"Hello\nworld",
        b"  spaces  ",
        "héllo".encode("utf-8"),
        "🙂 emoji".encode("utf-8"),
        b"<|endoftext|>",
        b"a\tb\r\nc",
        b"\x00\x01\x02 binary-ish \x7f",
        "日本語のテスト".encode("utf-8"),
        b"x" * 5000,
        b"",
        b" ".join(f"word{index}".encode("ascii") for index in range(200)),
        # RWKV chat contract markers are ordinary multi-token text, not
        # single vocabulary entries.
        b"User: Assistant: System: <think> </think>\n\n",
    ]
    # Every byte value, in a single buffer, exercises the whole ByteLevel map.
    cases.append(bytes(range(256)))
    return cases


def _check_generated_api(handle, reference) -> None:
    """Exercise emitted model APIs, not only standalone tokenizer entry points."""
    from build_ir_v8 import _generate_tokenizer_c_code

    api = _generate_tokenizer_c_code("rwkv_world_trie", len(reference.raw_by_id), 0)
    source = """
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "tokenizer/tokenizer.h"
#define CK_EXPORT
#define MAX_SEQ_LEN 8
#define A_TOKEN_IDS 0
static int32_t token_buffer[MAX_SEQ_LEN];
static struct { CKTokenizer *tokenizer; unsigned char *bump; } model;
static __typeof__(model) *g_model = &model;
void install_tokenizer(CKTokenizer *tokenizer) {
    model.tokenizer = tokenizer;
    model.bump = (unsigned char *)token_buffer;
}
""" + api["api_functions"]
    with tempfile.TemporaryDirectory(prefix="rwkv-tokenizer-api-") as directory:
        path = Path(directory) / "api.c"
        library = Path(directory) / "api.so"
        path.write_text(source)
        subprocess.run(["cc", "-shared", "-fPIC", "-std=gnu11", "-Werror",
                        "-I", str(ROOT / "include"), str(path), str(LIB_PATH),
                        "-Wl,-rpath," + str(LIB_PATH.parent), "-o", str(library)], check=True)
        model = ctypes.CDLL(str(library))
        model.install_tokenizer.argtypes = [ctypes.c_void_p]
        model.install_tokenizer(handle)
        for name in ("ck_model_encode_text", "ck_model_count_text_tokens"):
            getattr(model, name).argtypes = [ctypes.c_char_p, ctypes.c_int]
            getattr(model, name).restype = ctypes.c_int
        model.ck_model_lookup_token.argtypes = [ctypes.c_char_p]
        model.ck_model_lookup_token.restype = ctypes.c_int32
        short = bytes(range(8))
        long = bytes(range(256))
        assert model.ck_model_count_text_tokens(long, len(long)) == len(reference.encode(long))
        assert model.ck_model_encode_text(long, len(long)) == -1
        assert model.ck_model_encode_text(short, len(short)) == len(reference.encode(short))
        for raw in (b" ", b"\n", b"\xff"):
            assert model.ck_model_lookup_token(raw) == reference._terminal[raw]
        assert model.ck_model_lookup_token(b"not-a-vocabulary-piece-123456789") == -1


def main() -> int:
    tokenizer_json = _find_tokenizer_json()
    if tokenizer_json is None:
        print("SKIP: no RWKV World tokenizer.json found")
        return 0

    from convert_gguf_to_bump_v8 import load_tokenizer_json  # type: ignore

    document = json.loads(tokenizer_json.read_text(encoding="utf-8"))
    vocabulary = document["model"]["vocab"]
    vocab_size = len(vocabulary)
    offsets, strings, _merges, _scores, _types = load_tokenizer_json(
        str(tokenizer_json), vocab_size
    )

    lib = _load_library()
    handle = lib.ck_tokenizer_create(CK_TOKENIZER_RWKV_WORLD_TRIE)
    if not handle:
        raise SystemExit("ck_tokenizer_create failed")
    try:
        lib.ck_tokenizer_set_use_trie(handle, True)
        lib.ck_tokenizer_set_add_bos_eos(handle, False, False)
        offset_array = (ctypes.c_int32 * len(offsets))(*offsets)
        if lib.ck_tokenizer_load_binary(
            handle, vocab_size, offset_array, strings, 0, None
        ) != 0:
            raise SystemExit("ck_tokenizer_load_binary failed")
        lib.ck_tokenizer_set_special_ids(handle, 0, -1, 0, 0, -1)

        def encode(data: bytes) -> list[int]:
            capacity = max(len(data) + 2, 2)
            buffer = (ctypes.c_int32 * capacity)()
            count = lib.ck_tokenizer_encode(handle, data, len(data), buffer, capacity)
            return list(buffer[:count])

        def decode(ids: list[int]) -> bytes:
            id_array = (ctypes.c_int32 * len(ids))(*ids)
            buffer = ctypes.create_string_buffer(len(ids) * 256 + 64)
            count = lib.ck_tokenizer_decode(
                handle, id_array, len(ids), buffer, len(buffer)
            )
            return buffer.raw[:count]

        reference = ReferenceTokenizer(vocabulary)
        failures = 0
        for data in _prompt_cases():
            expected = reference.encode(data)
            actual = encode(data)
            if expected != actual:
                failures += 1
                print(f"ENCODE MISMATCH len={len(data)} data={data[:40]!r}")
                for index, (want, got) in enumerate(zip(expected, actual)):
                    if want != got:
                        print(f"  token {index}: expected {want}, got {got}")
                        break
                print(f"  lengths expected={len(expected)} actual={len(actual)}")
                continue
            decoded = decode(actual)
            if decoded != data:
                failures += 1
                print(f"DECODE MISMATCH len={len(data)} data={data[:40]!r}")
                print(f"  got {decoded[:60]!r}")

        # Capacity failure invalidates the entire prefix, and must not poison
        # a later encode on the same tokenizer.
        data = bytes(range(256)) * 3
        expected = reference.encode(data)
        exact = (ctypes.c_int32 * len(expected))()
        assert lib.ck_tokenizer_encode(handle, data, len(data), exact, len(expected)) == len(expected)
        assert list(exact) == expected
        assert lib.ck_tokenizer_encode(handle, data, len(data), exact, len(expected) - 1) == -1
        assert encode(data) == expected
        assert lib.ck_tokenizer_lookup(handle, b"not-a-vocabulary-piece-123456789") == -1
        # Unlike stored ByteLevel vocabulary strings, lookup accepts raw bytes.
        for raw in (b" ", b"\n", b"\xff"):
            assert lib.ck_tokenizer_lookup(handle, raw) == reference._terminal[raw]

        if lib.ck_tokenizer_lookup(handle, b"<|endoftext|>") != 0:
            failures += 1
            print("LOOKUP MISMATCH: <|endoftext|> did not resolve to id 0")

        sparse = lib.ck_tokenizer_create(CK_TOKENIZER_RWKV_WORLD_TRIE)
        assert sparse
        try:
            lib.ck_tokenizer_set_use_trie(sparse, True)
            offsets = (ctypes.c_int32 * 2)(0, 2)
            assert lib.ck_tokenizer_load_binary(sparse, 1, offsets, b"a\0", 0, None) == 0
            target = (ctypes.c_int32 * 8)()
            assert lib.ck_tokenizer_encode(sparse, b"ab", 2, target, 8) == -1
            assert lib.ck_tokenizer_encode(sparse, b"a", 1, target, 8) == 1
        finally:
            lib.ck_tokenizer_free(sparse)

        lib.ck_tokenizer_add_token.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int32, ctypes.c_float]
        lib.ck_tokenizer_add_token.restype = ctypes.c_int
        assert lib.ck_tokenizer_add_token(handle, b"x<|endoftext|>y", vocab_size, 0.0) == 0
        assert encode(b"x<|endoftext|>y") == reference.encode(b"x<|endoftext|>y")
        assert lib.ck_tokenizer_encode(handle, b"x", 1, exact, 0) == -1

        _check_generated_api(handle, reference)

        # The publisher's class is the strongest oracle when it is available.
        reference_path = tokenizer_json.with_name("tokenization_rwkv7.py")
        reference_class = (
            _load_reference_class(reference_path)
            if reference_path.is_file()
            else None
        )
        if reference_class is not None:
            oracle = reference_class(
                vocab_file=None, tokenizer_file=str(tokenizer_json)
            )
            for text in (
                "Hello world",
                "Hello\nworld",
                "  spaces  ",
                "héllo",
                "🙂 emoji",
                "<|endoftext|>",
                "日本語のテスト",
            ):
                expected = oracle.encode(text)
                actual = encode(text.encode("utf-8"))
                if expected != actual:
                    failures += 1
                    print(f"HF ORACLE MISMATCH {text!r}")
                    print(f"  expected {expected}")
                    print(f"  actual   {actual}")
                decoded = decode(actual)
                if decoded != text.encode("utf-8"):
                    failures += 1
                    print(f"HF DECODE ROUNDTRIP MISMATCH {text!r}: {decoded[:60]!r}")
        else:
            print("Note: publisher Rwkv7Tokenizer unavailable; Python reference only")
    finally:
        lib.ck_tokenizer_free(handle)

    print(
        f"{tokenizer_json}: vocab={vocab_size} "
        f"cases={len(_prompt_cases())} failures={failures}"
    )
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

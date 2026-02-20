#!/usr/bin/env python3
"""Determinism + roundtrip gate for CK true_bpe/ascii_bpe artifacts."""

from __future__ import annotations

import argparse
import ctypes
import json
import struct
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def _load_runtime(lib_path: Path):
    lib = ctypes.CDLL(str(lib_path))
    lib.ck_true_bpe_create.restype = ctypes.c_void_p
    lib.ck_true_bpe_free.argtypes = [ctypes.c_void_p]
    lib.ck_true_bpe_load_binary.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int32),
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int32),
    ]
    lib.ck_true_bpe_load_binary.restype = ctypes.c_int
    lib.ck_true_bpe_encode.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int32),
        ctypes.c_int,
    ]
    lib.ck_true_bpe_encode.restype = ctypes.c_int
    lib.ck_true_bpe_decode.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_int32),
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
    ]
    lib.ck_true_bpe_decode.restype = ctypes.c_int
    return lib


def _resolve_bin_dir(run_dir: Path | None, bin_dir: Path | None) -> Path:
    if bin_dir is not None:
        return bin_dir
    if run_dir is None:
        raise SystemExit("ERROR: provide --run or --bin-dir")
    cand = run_dir / "tokenizer_bin"
    if cand.exists():
        return cand
    raise SystemExit(f"ERROR: tokenizer_bin not found under run dir: {run_dir}")


def _load_artifacts(bin_dir: Path):
    meta_path = bin_dir / "tokenizer_meta.json"
    offsets_path = bin_dir / "vocab_offsets.bin"
    strings_path = bin_dir / "vocab_strings.bin"
    merges_path = bin_dir / "vocab_merges.bin"
    required = [meta_path, offsets_path, strings_path, merges_path]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise SystemExit("ERROR: missing tokenizer artifacts:\n  " + "\n  ".join(missing))

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    vocab_size = int(meta.get("vocab_size") or 0)
    num_merges = int(meta.get("num_merges") or 0)
    mode = str(meta.get("mode") or "unknown")
    if vocab_size <= 0 or num_merges < 0:
        raise SystemExit(f"ERROR: invalid tokenizer_meta values: vocab_size={vocab_size}, num_merges={num_merges}")

    offsets_b = offsets_path.read_bytes()
    merges_b = merges_path.read_bytes()
    strings_b = strings_path.read_bytes()
    if len(offsets_b) != vocab_size * 4:
        raise SystemExit(f"ERROR: bad offsets size: {len(offsets_b)}")
    if len(merges_b) != num_merges * 3 * 4:
        raise SystemExit(f"ERROR: bad merges size: {len(merges_b)}")

    offsets = list(struct.unpack("<" + ("i" * vocab_size), offsets_b))
    merges = list(struct.unpack("<" + ("i" * (num_merges * 3)), merges_b)) if num_merges > 0 else []
    return mode, vocab_size, num_merges, offsets, merges, strings_b


def _encode(lib, bpe, payload: bytes) -> list[int]:
    max_ids = max(4096, len(payload) * 8)
    out = (ctypes.c_int32 * max_ids)()
    n = int(lib.ck_true_bpe_encode(bpe, payload, len(payload), out, max_ids))
    if n < 0:
        raise SystemExit(f"ERROR: ck_true_bpe_encode returned {n}")
    return [int(out[i]) for i in range(n)]


def _decode(lib, bpe, ids: list[int], hint_bytes: int) -> bytes:
    arr = (ctypes.c_int32 * len(ids))(*ids)
    cap = max(1024, hint_bytes * 4 + 1024)
    out = ctypes.create_string_buffer(cap)
    n = int(lib.ck_true_bpe_decode(bpe, arr, len(ids), out, cap))
    if n < 0:
        raise SystemExit(f"ERROR: ck_true_bpe_decode returned {n}")
    if n >= cap - 1:
        cap = max(cap * 4, 65536)
        out = ctypes.create_string_buffer(cap)
        n = int(lib.ck_true_bpe_decode(bpe, arr, len(ids), out, cap))
        if n < 0:
            raise SystemExit(f"ERROR: ck_true_bpe_decode retry returned {n}")
    return out.raw[:n]


def main() -> int:
    ap = argparse.ArgumentParser(description="Roundtrip determinism gate for CK true_bpe/ascii_bpe")
    ap.add_argument("--run", default=None, help="Run dir containing tokenizer_bin")
    ap.add_argument("--bin-dir", default=None, help="Direct tokenizer_bin path")
    ap.add_argument("--dataset", required=True, help="Dataset text path to verify")
    ap.add_argument("--lib", default=str(ROOT / "build" / "libckernel_tokenizer.so"), help="libckernel_tokenizer.so path")
    ap.add_argument("--require-ascii", action="store_true", help="Fail if dataset contains non-ASCII bytes")
    args = ap.parse_args()

    run_dir = Path(args.run).expanduser().resolve() if args.run else None
    bin_dir = _resolve_bin_dir(run_dir, Path(args.bin_dir).expanduser().resolve() if args.bin_dir else None)
    lib_path = Path(args.lib).expanduser().resolve()
    dataset_path = Path(args.dataset).expanduser().resolve()

    if not lib_path.exists():
        raise SystemExit(f"ERROR: tokenizer library not found: {lib_path}")
    if not dataset_path.exists():
        raise SystemExit(f"ERROR: dataset not found: {dataset_path}")

    payload = dataset_path.read_bytes()
    if b"\x00" in payload:
        raise SystemExit("ERROR: dataset contains NUL byte; unsupported for this roundtrip gate")
    if args.require_ascii and any(b >= 128 for b in payload):
        raise SystemExit("ERROR: dataset contains non-ASCII bytes but --require-ascii was set")

    mode, vocab_size, num_merges, offsets, merges, strings_b = _load_artifacts(bin_dir)
    if mode != "ascii_bpe":
        raise SystemExit(f"ERROR: tokenizer mode is {mode}; this gate is only for ascii_bpe")
    lib = _load_runtime(lib_path)
    bpe = lib.ck_true_bpe_create()
    if not bpe:
        raise SystemExit("ERROR: ck_true_bpe_create failed")

    offsets_arr = (ctypes.c_int32 * vocab_size)(*offsets)
    merges_arr = (ctypes.c_int32 * (num_merges * 3))(*merges)
    strings_buf = ctypes.create_string_buffer(strings_b + b"\x00")

    try:
        rc = lib.ck_true_bpe_load_binary(
            bpe,
            vocab_size,
            offsets_arr,
            ctypes.cast(strings_buf, ctypes.c_char_p),
            num_merges,
            merges_arr,
        )
        if rc != 0:
            raise SystemExit(f"ERROR: ck_true_bpe_load_binary failed rc={rc}")

        ids1 = _encode(lib, bpe, payload)
        ids2 = _encode(lib, bpe, payload)
        if ids1 != ids2:
            raise SystemExit("ERROR: encode is non-deterministic (ids mismatch across runs)")

        decoded = _decode(lib, bpe, ids1, len(payload))
        if decoded != payload:
            mismatch = next((i for i, (a, b) in enumerate(zip(payload, decoded)) if a != b), None)
            if mismatch is None:
                mismatch = min(len(payload), len(decoded))
            raise SystemExit(
                "ERROR: roundtrip mismatch\n"
                f"  mode={mode}\n"
                f"  original_bytes={len(payload)} decoded_bytes={len(decoded)}\n"
                f"  first_mismatch_at={mismatch}"
            )

        print("ascii_bpe roundtrip PASS")
        print(f"  mode:         {mode}")
        print(f"  dataset:      {dataset_path}")
        print(f"  tokenizer:    {bin_dir}")
        print(f"  vocab_size:   {vocab_size}")
        print(f"  merges:       {num_merges}")
        print(f"  token_count:  {len(ids1)}")
        print(f"  bytes:        {len(payload)}")
        return 0
    finally:
        lib.ck_true_bpe_free(bpe)


if __name__ == "__main__":
    raise SystemExit(main())

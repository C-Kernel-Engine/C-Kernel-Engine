#!/usr/bin/env python3
"""
Pack row-structured training data into fixed-length token windows.

Rules:
1) Each non-empty input row is encoded independently and wrapped as:
     [BOS] + row_tokens + [EOS]
2) No row is split across windows.
3) Rows with token length > seq_len are dropped (reported).
4) Windows are padded with PAD token to exactly seq_len.

Outputs:
- Integer token stream file (one token id per line)
- JSON report with packing stats and recommended total_tokens
"""

from __future__ import annotations

import argparse
import ctypes
import json
import struct
import math
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_rows(path: Path) -> list[str]:
    try:
        raw = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        raw = path.read_text(encoding="utf-8", errors="ignore")
    out: list[str] = []
    for line in raw.splitlines():
        row = line.strip()
        if not row:
            continue
        out.append(row)
    return out


def _resolve_special_ids(
    tokenizer_json: Path,
    bos_override: int | None,
    eos_override: int | None,
    pad_override: int | None,
) -> tuple[int, int, int]:
    if not tokenizer_json.exists():
        raise SystemExit(f"ERROR: tokenizer.json not found: {tokenizer_json}")
    doc = _load_json(tokenizer_json)

    def _from_added(name: str) -> int | None:
        added = doc.get("added_tokens")
        if not isinstance(added, list):
            return None
        for row in added:
            if not isinstance(row, dict):
                continue
            if str(row.get("content") or "") != name:
                continue
            tid = row.get("id")
            if isinstance(tid, int) and tid >= 0:
                return int(tid)
        return None

    def _from_vocab(name: str) -> int | None:
        model = doc.get("model")
        if not isinstance(model, dict):
            return None
        vocab = model.get("vocab")
        if not isinstance(vocab, dict):
            return None
        tid = vocab.get(name)
        if isinstance(tid, int) and tid >= 0:
            return int(tid)
        return None

    def _pick(*names: str, fallback: int) -> int:
        for n in names:
            v = _from_added(n)
            if v is not None:
                return v
        for n in names:
            v = _from_vocab(n)
            if v is not None:
                return v
        return int(fallback)

    bos = int(bos_override) if bos_override is not None else _pick("<|bos|>", "<s>", fallback=1)
    eos = int(eos_override) if eos_override is not None else _pick("<|eos|>", "</s>", "<eos>", fallback=2)
    pad = int(pad_override) if pad_override is not None else _pick("<|pad|>", "<pad>", fallback=3)
    return bos, eos, pad


def _load_true_bpe_runtime(lib_path: Path):
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
    return lib


def _load_true_bpe_binary_artifacts(bin_dir: Path):
    meta_path = bin_dir / "tokenizer_meta.json"
    offsets_path = bin_dir / "vocab_offsets.bin"
    strings_path = bin_dir / "vocab_strings.bin"
    merges_path = bin_dir / "vocab_merges.bin"
    required = [meta_path, offsets_path, strings_path, merges_path]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise RuntimeError(
            "BPE binary artifacts missing. Expected files:\n  " + "\n  ".join(missing)
        )
    meta = _load_json(meta_path)
    vocab_size = int(meta.get("vocab_size") or 0)
    num_merges = int(meta.get("num_merges") or 0)
    if vocab_size <= 0:
        raise RuntimeError(f"Invalid vocab_size in {meta_path}: {vocab_size}")
    if num_merges < 0:
        raise RuntimeError(f"Invalid num_merges in {meta_path}: {num_merges}")

    offsets_b = offsets_path.read_bytes()
    merges_b = merges_path.read_bytes()
    strings_b = strings_path.read_bytes()
    expected_offsets_bytes = vocab_size * 4
    expected_merges_bytes = num_merges * 3 * 4
    if len(offsets_b) != expected_offsets_bytes:
        raise RuntimeError(
            f"Invalid offsets size in {offsets_path}: got {len(offsets_b)}, expected {expected_offsets_bytes}"
        )
    if len(merges_b) != expected_merges_bytes:
        raise RuntimeError(
            f"Invalid merges size in {merges_path}: got {len(merges_b)}, expected {expected_merges_bytes}"
        )
    offsets = list(struct.unpack("<" + ("i" * vocab_size), offsets_b))
    merges = list(struct.unpack("<" + ("i" * (num_merges * 3)), merges_b)) if num_merges > 0 else []
    offsets_arr = (ctypes.c_int32 * vocab_size)(*offsets)
    merges_arr = (ctypes.c_int32 * (num_merges * 3))(*merges)
    strings_buf = ctypes.create_string_buffer(strings_b + b"\x00")
    return vocab_size, num_merges, offsets_arr, merges_arr, strings_buf


class _TrueBPEHandle:
    def __init__(self, tokenizer_lib: Path, bpe_bin_dir: Path):
        if not tokenizer_lib.exists():
            raise RuntimeError(f"Tokenizer library not found: {tokenizer_lib}")
        if not bpe_bin_dir.exists():
            raise RuntimeError(f"Tokenizer binary dir not found: {bpe_bin_dir}")
        self.lib = _load_true_bpe_runtime(tokenizer_lib)
        self.bpe = self.lib.ck_true_bpe_create()
        if not self.bpe:
            raise RuntimeError("ck_true_bpe_create failed")
        (
            self.vocab_size,
            self.num_merges,
            self.offsets_arr,
            self.merges_arr,
            self.strings_buf,
        ) = _load_true_bpe_binary_artifacts(bpe_bin_dir)
        rc = self.lib.ck_true_bpe_load_binary(
            self.bpe,
            self.vocab_size,
            self.offsets_arr,
            ctypes.cast(self.strings_buf, ctypes.c_char_p),
            self.num_merges,
            self.merges_arr,
        )
        if rc != 0:
            raise RuntimeError(f"ck_true_bpe_load_binary failed rc={rc}")

    def close(self) -> None:
        if getattr(self, "bpe", None):
            self.lib.ck_true_bpe_free(self.bpe)
            self.bpe = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def encode(self, text: str) -> list[int]:
        text_bytes = (text or "").encode("utf-8")
        max_ids = max(256, len(text_bytes) * 8)
        out = (ctypes.c_int32 * max_ids)()
        n = int(self.lib.ck_true_bpe_encode(self.bpe, text_bytes, -1, out, max_ids))
        if n < 0:
            raise RuntimeError(f"ck_true_bpe_encode failed rc={n}")
        return [int(out[i]) for i in range(n)]


def _encode_row_with_fallback(handle: _TrueBPEHandle, row: str, chunk_chars: int = 4096) -> list[int]:
    if not row:
        return []
    ids = handle.encode(row)
    if len(ids) > 0:
        return ids
    if len(row) <= 1:
        raise RuntimeError("BPE encoding produced 0 ids for non-empty row.")
    out: list[int] = []
    step = max(1, min(int(chunk_chars), len(row) // 2))
    for i in range(0, len(row), step):
        chunk = row[i : i + step]
        if not chunk:
            continue
        cid = handle.encode(chunk)
        if len(cid) > 0:
            out.extend(cid)
            continue
        if len(chunk) <= 1:
            raise RuntimeError("BPE encoding produced 0 ids for fallback chunk.")
        half = max(1, len(chunk) // 2)
        for j in range(0, len(chunk), half):
            piece = chunk[j : j + half]
            if not piece:
                continue
            pid = handle.encode(piece)
            if len(pid) == 0:
                raise RuntimeError("BPE encoding produced 0 ids for fallback piece.")
            out.extend(pid)
    return out


def _pack_rows(
    row_token_ids: list[list[int]],
    seq_len: int,
    pad_id: int,
) -> tuple[list[int], dict[str, Any]]:
    seq_len_i = max(1, int(seq_len))
    pad_i = int(pad_id)
    packed: list[int] = []
    current: list[int] = []
    windows = 0
    pad_tokens = 0
    non_pad_tokens = 0
    for row_ids in row_token_ids:
        n = len(row_ids)
        if n <= 0:
            continue
        if len(current) + n <= seq_len_i:
            current.extend(row_ids)
            non_pad_tokens += n
            continue
        if current:
            pad_n = seq_len_i - len(current)
            if pad_n > 0:
                current.extend([pad_i] * pad_n)
                pad_tokens += pad_n
            packed.extend(current)
            windows += 1
        current = list(row_ids)
        non_pad_tokens += n
    if current:
        pad_n = seq_len_i - len(current)
        if pad_n > 0:
            current.extend([pad_i] * pad_n)
            pad_tokens += pad_n
        packed.extend(current)
        windows += 1
    packed_window_tokens = int(windows * seq_len_i)
    token_file_tokens = int(len(packed) + 1)  # +1 trailer token for x/y shift without stream repeat
    stats = {
        "windows": int(windows),
        "seq_len": int(seq_len_i),
        "packed_window_tokens": packed_window_tokens,
        "token_file_token_count": token_file_tokens,
        "non_pad_tokens": int(non_pad_tokens),
        "pad_tokens": int(pad_tokens),
        "fill_ratio": float(non_pad_tokens) / float(max(1, packed_window_tokens)),
        "recommended_total_tokens": packed_window_tokens,
        "steps_per_epoch": int(windows),
        "pad_token_id": int(pad_i),
    }
    return packed, stats


def _percentile_int(values: list[int], q: float) -> int:
    if not values:
        return 0
    v = sorted(int(x) for x in values)
    if len(v) == 1:
        return int(v[0])
    qq = max(0.0, min(1.0, float(q)))
    idx = int(math.ceil(qq * len(v)) - 1)
    idx = max(0, min(idx, len(v) - 1))
    return int(v[idx])


def main() -> int:
    ap = argparse.ArgumentParser(description="Pack row-structured samples into fixed token windows")
    ap.add_argument("--dataset", required=True, help="Input UTF-8 dataset text file (one sample per line)")
    ap.add_argument("--tokenizer-lib", default="build/libckernel_tokenizer.so", help="Path to libckernel_tokenizer.so")
    ap.add_argument("--tokenizer-bin", required=True, help="Path to tokenizer_bin directory")
    ap.add_argument("--tokenizer-json", required=True, help="Path to tokenizer.json")
    ap.add_argument("--seq-len", type=int, required=True, help="Context length/window size")
    ap.add_argument("--out", required=True, help="Output token stream file (one id per line)")
    ap.add_argument("--report-json", default=None, help="Optional report JSON path")
    ap.add_argument("--bos-id", type=int, default=None, help="Override BOS token id")
    ap.add_argument("--eos-id", type=int, default=None, help="Override EOS token id")
    ap.add_argument("--pad-id", type=int, default=None, help="Override PAD token id")
    args = ap.parse_args()

    dataset = Path(args.dataset).expanduser().resolve()
    tokenizer_lib = Path(args.tokenizer_lib).expanduser().resolve()
    tokenizer_bin = Path(args.tokenizer_bin).expanduser().resolve()
    tokenizer_json = Path(args.tokenizer_json).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()
    report_path = Path(args.report_json).expanduser().resolve() if args.report_json else None

    if int(args.seq_len) < 1:
        raise SystemExit(f"ERROR: --seq-len must be >= 1, got {args.seq_len}")
    if not dataset.exists():
        raise SystemExit(f"ERROR: dataset not found: {dataset}")

    rows = _read_rows(dataset)
    if not rows:
        raise SystemExit(f"ERROR: no non-empty rows in dataset: {dataset}")

    bos_id, eos_id, pad_id = _resolve_special_ids(
        tokenizer_json=tokenizer_json,
        bos_override=args.bos_id,
        eos_override=args.eos_id,
        pad_override=args.pad_id,
    )

    dropped_rows: list[dict[str, Any]] = []
    row_payloads: list[list[int]] = []
    kept_row_lengths: list[int] = []
    min_row = None
    max_row = 0

    with _TrueBPEHandle(tokenizer_lib, tokenizer_bin) as handle:
        for idx, row in enumerate(rows, start=1):
            rid = _encode_row_with_fallback(handle, row)
            full = [int(bos_id), *[int(v) for v in rid], int(eos_id)]
            n = len(full)
            max_row = max(max_row, n)
            min_row = n if min_row is None else min(min_row, n)
            if n > int(args.seq_len):
                if len(dropped_rows) < 16:
                    dropped_rows.append(
                        {"row_index": int(idx), "token_count": int(n), "preview": row[:160]}
                    )
                continue
            row_payloads.append(full)
            kept_row_lengths.append(int(n))

    if not row_payloads:
        raise SystemExit(
            "ERROR: all rows exceed seq_len after BOS/EOS wrap; "
            "increase --seq-len or shorten samples"
        )

    packed_ids, stats = _pack_rows(row_payloads, int(args.seq_len), int(pad_id))
    packed_ids_out = list(packed_ids)
    packed_ids_out.append(int(pad_id))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(str(v) for v in packed_ids_out) + "\n", encoding="utf-8")

    report = {
        "schema": "ck.training_pack.v1",
        "dataset": str(dataset),
        "tokenizer_json": str(tokenizer_json),
        "tokenizer_bin": str(tokenizer_bin),
        "ids": {"bos": int(bos_id), "eos": int(eos_id), "pad": int(pad_id)},
        "rows_total": int(len(rows)),
        "rows_kept": int(len(row_payloads)),
        "rows_dropped_oversize": int(len(rows) - len(row_payloads)),
        "row_tokens_min": int(min_row or 0),
        "row_tokens_max": int(max_row),
        "row_tokens_p50": int(_percentile_int(kept_row_lengths, 0.50)),
        "row_tokens_p90": int(_percentile_int(kept_row_lengths, 0.90)),
        "row_tokens_p95": int(_percentile_int(kept_row_lengths, 0.95)),
        "row_tokens_p99": int(_percentile_int(kept_row_lengths, 0.99)),
        "stats": stats,
        "dropped_examples": dropped_rows,
        "token_file": str(out_path),
    }
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(
        "[pack] "
        f"rows={report['rows_total']} kept={report['rows_kept']} dropped={report['rows_dropped_oversize']} "
        f"windows={stats['windows']} seq_len={stats['seq_len']} "
        f"fill={float(stats['fill_ratio']):.4f} "
        f"recommended_total_tokens={stats['recommended_total_tokens']} "
        f"token_file_tokens={stats['token_file_token_count']}"
    )
    if report["rows_dropped_oversize"] > 0:
        print("[pack] dropped oversized rows; inspect report_json for samples.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

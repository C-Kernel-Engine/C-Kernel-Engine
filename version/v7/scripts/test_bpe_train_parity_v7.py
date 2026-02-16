#!/usr/bin/env python3
"""Parity gate for ck-bpe-train output (HF tokenizer vs CK true_bpe runtime).

Flow:
1) Build a tiny corpus (or use user corpus dir)
2) Run build/ck-bpe-train to produce tokenizer.json + binary artifacts
3) Load tokenizer.json with HuggingFace `tokenizers`
4) Load binary artifacts with CK `ck_true_bpe_load_binary`
5) Compare token IDs on a deterministic text set
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import shlex
import struct
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_TEXTS = [
    "hello world",
    "the quick brown fox jumps over the lazy dog",
    "svg path d equals M 10 10 L 20 20",
    "optimizer state m and v buffers",
    "residual add and rmsnorm backward",
]


def _require_tokenizers() -> None:
    try:
        from tokenizers import Tokenizer  # noqa: F401
    except Exception as exc:
        raise SystemExit(
            "ERROR: Python package `tokenizers` is required for HF parity checks.\n"
            "Install in .venv: .venv/bin/pip install tokenizers"
        ) from exc


def _write_default_corpus(corpus_dir: Path) -> list[str]:
    corpus_dir.mkdir(parents=True, exist_ok=True)
    a = corpus_dir / "sample_a.txt"
    b = corpus_dir / "sample_b.txt"

    a.write_text(
        "\n".join(
            [
                "hello world hello world",
                "this is a dummy corpus for bpe",
                "residual add keeps skip connections stable",
                "gradient clipping happens before optimizer update",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    b.write_text(
        "\n".join(
            [
                "hello there general kenobi",
                "bpe tokenizer training test",
                "attention qkv projection and softmax",
                "memory canary detects overflow",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return [
        "hello world",
        "this is a dummy corpus for bpe",
        "hello there general kenobi",
        "bpe tokenizer training test",
        "HELLO world!",
    ]


def _collect_eval_texts(corpus_dir: Path, max_texts: int) -> list[str]:
    texts: list[str] = []
    seen: set[str] = set()
    exts = {".txt", ".md", ".svg", ".xml", ".html", ".json"}

    for p in sorted(corpus_dir.rglob("*")):
        if not p.is_file():
            continue
        if p.suffix.lower() not in exts:
            continue
        try:
            data = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        for raw in data.splitlines():
            line = raw.strip()
            if not line:
                continue
            if len(line) > 240:
                line = line[:240]
            if line in seen:
                continue
            seen.add(line)
            texts.append(line)
            if len(texts) >= max_texts:
                break
        if len(texts) >= max_texts:
            break

    for t in DEFAULT_TEXTS:
        if t not in seen:
            texts.append(t)
            seen.add(t)

    return texts[:max_texts]


def _run(cmd: list[str], cwd: Path) -> None:
    print("[run]", " ".join(shlex.quote(c) for c in cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)


def _load_ck_runtime(lib_path: Path):
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


def _load_binary_artifacts(bin_dir: Path):
    meta = json.loads((bin_dir / "tokenizer_meta.json").read_text(encoding="utf-8"))
    vocab_size = int(meta["vocab_size"])
    num_merges = int(meta["num_merges"])

    offsets_b = (bin_dir / "vocab_offsets.bin").read_bytes()
    merges_b = (bin_dir / "vocab_merges.bin").read_bytes()
    strings_b = (bin_dir / "vocab_strings.bin").read_bytes()

    offsets = list(struct.unpack("<" + "i" * vocab_size, offsets_b))
    merges = list(struct.unpack("<" + "i" * (num_merges * 3), merges_b))

    OffArr = (ctypes.c_int32 * vocab_size)(*offsets)
    MergeArr = (ctypes.c_int32 * (num_merges * 3))(*merges)
    StrBuf = ctypes.create_string_buffer(strings_b + b"\x00")
    return vocab_size, num_merges, OffArr, MergeArr, StrBuf


def main() -> int:
    ap = argparse.ArgumentParser(description="v7 BPE trainer parity gate")
    ap.add_argument("--corpus-dir", default=None, help="Optional corpus dir (default: generated dummy corpus)")
    ap.add_argument("--work-dir", default=None, help="Optional work dir for artifacts")
    ap.add_argument("--vocab-size", type=int, default=320)
    ap.add_argument("--min-freq", type=int, default=2)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--max-texts", type=int, default=24)
    ap.add_argument("--json-out", default=None, help="Optional JSON report output")
    args = ap.parse_args()

    _require_tokenizers()
    from tokenizers import Tokenizer

    bpe_bin = ROOT / "build" / "ck-bpe-train"
    lib_path = ROOT / "build" / "libckernel_tokenizer.so"
    if not bpe_bin.exists():
        raise SystemExit(f"ERROR: missing {bpe_bin}. Run `make ck-bpe-train` first.")
    if not lib_path.exists():
        raise SystemExit(f"ERROR: missing {lib_path}. Run `make tokenizer` first.")

    if args.work_dir:
        work_dir = Path(args.work_dir).resolve()
        work_dir.mkdir(parents=True, exist_ok=True)
        cleanup = False
    else:
        tmp = tempfile.TemporaryDirectory(prefix="ck_v7_bpe_parity_")
        work_dir = Path(tmp.name)
        cleanup = True

    corpus_dir = Path(args.corpus_dir).resolve() if args.corpus_dir else (work_dir / "corpus")
    if args.corpus_dir:
        if not corpus_dir.exists():
            raise SystemExit(f"ERROR: corpus dir not found: {corpus_dir}")
        eval_texts = _collect_eval_texts(corpus_dir, args.max_texts)
    else:
        eval_texts = _write_default_corpus(corpus_dir)

    if not eval_texts:
        raise SystemExit("ERROR: no evaluation texts collected")

    out_json = work_dir / "tokenizer.json"
    out_bin = work_dir / "bin"
    out_bin.mkdir(parents=True, exist_ok=True)

    _run(
        [
            str(bpe_bin),
            "--corpus-dir",
            str(corpus_dir),
            "--out",
            str(out_json),
            "--binary-out-dir",
            str(out_bin),
            "--vocab-size",
            str(args.vocab_size),
            "--min-freq",
            str(args.min_freq),
            "--threads",
            str(args.threads),
        ],
        cwd=ROOT,
    )

    hf_tok = Tokenizer.from_file(str(out_json))
    lib = _load_ck_runtime(lib_path)
    vocab_size, num_merges, OffArr, MergeArr, StrBuf = _load_binary_artifacts(out_bin)

    bpe = lib.ck_true_bpe_create()
    if not bpe:
        raise SystemExit("ERROR: ck_true_bpe_create failed")

    rc = lib.ck_true_bpe_load_binary(
        bpe,
        vocab_size,
        OffArr,
        ctypes.cast(StrBuf, ctypes.c_char_p),
        num_merges,
        MergeArr,
    )
    if rc != 0:
        lib.ck_true_bpe_free(bpe)
        raise SystemExit(f"ERROR: ck_true_bpe_load_binary failed rc={rc}")

    mismatches = []
    compared = 0
    for text in eval_texts:
        hf_ids = hf_tok.encode(text).ids
        max_ids = max(4096, len(text.encode("utf-8")) * 8)
        out = (ctypes.c_int32 * max_ids)()
        n = lib.ck_true_bpe_encode(bpe, text.encode("utf-8"), -1, out, max_ids)
        ck_ids = list(out[:n])
        compared += 1
        if hf_ids != ck_ids:
            mismatches.append(
                {
                    "text": text,
                    "hf_ids": hf_ids,
                    "ck_ids": ck_ids,
                }
            )

    lib.ck_true_bpe_free(bpe)

    report = {
        "ok": len(mismatches) == 0,
        "compared": compared,
        "mismatch_count": len(mismatches),
        "corpus_dir": str(corpus_dir),
        "work_dir": str(work_dir),
        "tokenizer_json": str(out_json),
        "binary_dir": str(out_bin),
        "vocab_size": vocab_size,
        "num_merges": num_merges,
        "mismatches": mismatches[:5],
    }

    if args.json_out:
        out_path = Path(args.json_out).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"[out] {out_path}")

    if report["ok"]:
        print(f"PASS: v7 BPE trainer parity ({compared} texts)")
        if cleanup:
            # temp dir removed automatically by TemporaryDirectory GC
            pass
        return 0

    print(f"FAIL: v7 BPE trainer parity mismatches={len(mismatches)}/{compared}")
    for i, mm in enumerate(mismatches[:3], 1):
        print(f"  #{i} text={mm['text']!r}")
        print(f"      hf={mm['hf_ids'][:32]}")
        print(f"      ck={mm['ck_ids'][:32]}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

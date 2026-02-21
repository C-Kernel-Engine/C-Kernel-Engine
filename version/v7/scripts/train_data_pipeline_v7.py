#!/usr/bin/env python3
"""
High-level v7 training pipeline:
dataset -> (optional BPE) -> CK training -> optional monitoring helpers.

This is a convenience wrapper over existing v7 tools.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import importlib.util
import json
import math
import os
import re
import shutil
import shlex
import struct
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[3]
CK_RUN = ROOT / "version" / "v7" / "scripts" / "ck_run_v7.py"
TORCH_REF = ROOT / "version" / "v7" / "scripts" / "train_qwen3_torch_from_run_v7.py"
OPEN_VIS = ROOT / "version" / "v7" / "tools" / "open_ir_visualizer.py"
BPE_BIN = ROOT / "build" / "ck-bpe-train"
TOKENIZER_LIB = ROOT / "build" / "libckernel_tokenizer.so"
CK_CLI_BIN = ROOT / "build" / "ck-cli-v7"
CK_CHAT = ROOT / "scripts" / "ck_chat.py"

SVG_LINE = (
    '<svg width="100" height="100" xmlns="http://www.w3.org/2000/svg">'
    '<rect x="10" y="10" width="80" height="80" fill="red" stroke="black"/></svg>'
)


def _is_bpe_tokenizer_mode(tokenizer: str) -> bool:
    return tokenizer in {"bpe", "ascii_bpe"}


def _python_exec() -> str:
    venv_py = ROOT / ".venv" / "bin" / "python"
    return str(venv_py) if venv_py.exists() else sys.executable


def _run(cmd: list[str], cwd: Path) -> None:
    print("[run]", " ".join(shlex.quote(c) for c in cmd))
    result = subprocess.run(cmd, cwd=str(cwd), stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        stderr_msg = (result.stderr or "").strip()
        cmd_str = " ".join(shlex.quote(c) for c in cmd)
        msg = f"Command failed (exit {result.returncode}): {cmd_str}"
        if stderr_msg:
            msg += f"\n  stderr: {stderr_msg[-2000:]}"
        raise RuntimeError(msg)


def _run_capture(cmd: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    print("[run]", " ".join(shlex.quote(c) for c in cmd))
    result = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True)
    if result.returncode != 0:
        stderr_msg = (result.stderr or "").strip()
        cmd_str = " ".join(shlex.quote(c) for c in cmd)
        msg = f"Command failed (exit {result.returncode}): {cmd_str}"
        if stderr_msg:
            msg += f"\n  stderr: {stderr_msg[-2000:]}"
        raise RuntimeError(msg)
    return result


def _ensure_binary(path: Path, make_target: str) -> None:
    if path.exists():
        return
    _run(["make", "--no-print-directory", make_target], cwd=ROOT)
    if not path.exists():
        raise RuntimeError(f"expected binary after build: {path}")


def _write_svg_dataset(path: Path, repeats: int) -> None:
    lines = [SVG_LINE for _ in range(max(1, int(repeats)))]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _loss_stats(payload: dict[str, Any]) -> dict[str, Any]:
    curve = payload.get("loss_curve")
    if not isinstance(curve, list) or not curve:
        return {"steps": 0}
    vals: list[float] = []
    for row in curve:
        if not isinstance(row, dict):
            continue
        v = row.get("loss_ck", row.get("loss"))
        if isinstance(v, (int, float)) and math.isfinite(float(v)):
            vals.append(float(v))
    if not vals:
        return {"steps": 0, "note": "no_finite_losses"}
    min_idx = min(range(len(vals)), key=lambda i: vals[i])
    return {
        "steps": int(len(vals)),
        "first": float(vals[0]),
        "final": float(vals[-1]),
        "min": float(vals[min_idx]),
        "min_step": int(min_idx + 1),
    }


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_run_vocab_size(run_dir: Path) -> int | None:
    manifest = run_dir / "weights_manifest.json"
    if not manifest.exists():
        return None
    try:
        payload = _load_json(manifest)
    except Exception:
        return None
    cfg = payload.get("config") if isinstance(payload, dict) else None
    v = cfg.get("vocab_size") if isinstance(cfg, dict) else None
    if isinstance(v, int) and v > 0:
        return int(v)
    return None


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
    lib.ck_true_bpe_decode.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_int32),
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
    ]
    lib.ck_true_bpe_decode.restype = ctypes.c_int
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
            "BPE binary artifacts missing. Expected files:\n  "
            + "\n  ".join(missing)
        )

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
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
            f"Invalid offsets size in {offsets_path}: got {len(offsets_b)} bytes, "
            f"expected {expected_offsets_bytes}"
        )
    if len(merges_b) != expected_merges_bytes:
        raise RuntimeError(
            f"Invalid merges size in {merges_path}: got {len(merges_b)} bytes, "
            f"expected {expected_merges_bytes}"
        )

    offsets = list(struct.unpack("<" + ("i" * vocab_size), offsets_b))
    merges = list(struct.unpack("<" + ("i" * (num_merges * 3)), merges_b)) if num_merges > 0 else []

    offsets_arr = (ctypes.c_int32 * vocab_size)(*offsets)
    merges_arr = (ctypes.c_int32 * (num_merges * 3))(*merges)
    strings_buf = ctypes.create_string_buffer(strings_b + b"\x00")
    return vocab_size, num_merges, offsets_arr, merges_arr, strings_buf


class _TrueBPEHandle:
    def __init__(self, tokenizer_lib: Path, bin_dir: Path):
        self.lib = _load_true_bpe_runtime(tokenizer_lib)
        self.bpe = self.lib.ck_true_bpe_create()
        if not self.bpe:
            raise RuntimeError("ck_true_bpe_create failed")
        self._closed = False

        (
            self.vocab_size,
            self.num_merges,
            self.offsets_arr,
            self.merges_arr,
            self.strings_buf,
        ) = _load_true_bpe_binary_artifacts(bin_dir)
        rc = self.lib.ck_true_bpe_load_binary(
            self.bpe,
            self.vocab_size,
            self.offsets_arr,
            ctypes.cast(self.strings_buf, ctypes.c_char_p),
            self.num_merges,
            self.merges_arr,
        )
        if rc != 0:
            self.close()
            raise RuntimeError(f"ck_true_bpe_load_binary failed rc={rc}")

    def close(self) -> None:
        if self._closed:
            return
        self.lib.ck_true_bpe_free(self.bpe)
        self._closed = True

    def __enter__(self) -> "_TrueBPEHandle":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def encode(self, text: str) -> list[int]:
        text_bytes = text.encode("utf-8")
        max_ids = max(256, len(text_bytes) * 8)
        out = (ctypes.c_int32 * max_ids)()
        n = int(self.lib.ck_true_bpe_encode(self.bpe, text_bytes, -1, out, max_ids))
        if n < 0:
            raise RuntimeError(f"ck_true_bpe_encode failed rc={n}")
        return [int(out[i]) for i in range(n)]

    def decode(self, ids: list[int]) -> str:
        if not ids:
            return ""
        ids_arr = (ctypes.c_int32 * len(ids))(*[int(v) for v in ids])
        cap = max(4096, len(ids) * 16 + 1024)
        for _ in range(8):
            out = ctypes.create_string_buffer(cap)
            n = int(self.lib.ck_true_bpe_decode(self.bpe, ids_arr, len(ids), out, cap))
            if n < 0:
                raise RuntimeError(f"ck_true_bpe_decode failed rc={n}")
            if n < cap - 1:
                return out.raw[:n].decode("utf-8", errors="replace")
            cap *= 2
        raise RuntimeError("ck_true_bpe_decode exceeded buffer growth limit")


def _encode_large_text_with_bpe_handle(handle: _TrueBPEHandle, text: str, chunk_chars: int = 8192) -> list[int]:
    """
    Encode long corpora robustly.

    Some tokenizer runtimes can return 0 ids for very large single-buffer input.
    We preserve exact text bytes by encoding splitlines(keepends=True) and, for
    pathological long lines, sub-chunking by characters.
    """
    if not text:
        return []
    out: list[int] = []
    segments = text.splitlines(keepends=True)
    if not segments:
        segments = [text]
    for seg in segments:
        if not seg:
            continue
        ids = handle.encode(seg)
        if len(ids) > 0:
            out.extend(ids)
            continue
        # Fallback for very long segments.
        if len(seg) <= chunk_chars:
            continue
        for i in range(0, len(seg), chunk_chars):
            chunk = seg[i : i + chunk_chars]
            if not chunk:
                continue
            out.extend(handle.encode(chunk))
    return out


def _encode_with_ck_true_bpe(tokenizer_lib: Path, bin_dir: Path, text: str) -> list[int]:
    if not text:
        raise RuntimeError("BPE encoding requires non-empty training text.")
    with _TrueBPEHandle(tokenizer_lib, bin_dir) as handle:
        ids = _encode_large_text_with_bpe_handle(handle, text)
    if len(ids) <= 1:
        raise RuntimeError("BPE encoding produced <=1 token; provide richer data.")
    return ids


def _atomic_write_text(path: Path, content: str) -> None:
    """Write text to path atomically via temp file + rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        os.write(fd, content.encode("utf-8"))
        os.close(fd)
        fd = -1
        os.rename(tmp, str(path))
    except BaseException:
        if fd >= 0:
            os.close(fd)
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def _bytes_preview(raw: bytes, limit: int = 48) -> str:
    chunk = raw[:limit]
    text = "".join(chr(b) if 32 <= b <= 126 else "." for b in chunk)
    hx = " ".join(f"{b:02X}" for b in chunk)
    suffix = " ..." if len(raw) > limit else ""
    return f"text='{text}{suffix}' hex={hx}{suffix}"


def _validate_dataset_rows(
    dataset_path: Path,
    require_ascii: bool,
    require_svg_rows: bool,
    max_issues: int = 8,
) -> dict[str, Any]:
    payload = dataset_path.read_bytes()
    if not payload:
        raise SystemExit(f"ERROR: dataset is empty: {dataset_path}")
    if b"\x00" in payload:
        raise SystemExit(
            "ERROR: dataset contains NUL byte(s), unsupported for text training.\n"
            f"  path: {dataset_path}"
        )

    rows = payload.splitlines()
    non_empty = 0
    ascii_issues: list[tuple[int, int, int, bytes]] = []
    svg_issues: list[tuple[int, bytes]] = []

    for line_no, row in enumerate(rows, start=1):
        stripped = row.lstrip()
        if not stripped:
            continue
        non_empty += 1

        if require_ascii:
            bad_col = None
            bad_byte = None
            for col, byte in enumerate(row, start=1):
                if byte >= 128:
                    bad_col = col
                    bad_byte = byte
                    break
            if bad_col is not None and bad_byte is not None and len(ascii_issues) < max_issues:
                ascii_issues.append((line_no, bad_col, bad_byte, row))

        if require_svg_rows and (not stripped.startswith(b"<svg")) and len(svg_issues) < max_issues:
            svg_issues.append((line_no, row))

    if non_empty == 0:
        raise SystemExit(f"ERROR: dataset has no non-empty lines: {dataset_path}")

    if ascii_issues or svg_issues:
        msg: list[str] = [
            "ERROR: dataset validation failed",
            f"  path: {dataset_path}",
            f"  checks: require_ascii={bool(require_ascii)} require_svg_rows={bool(require_svg_rows)}",
        ]
        if ascii_issues:
            msg.append(f"  non_ascii_lines: {len(ascii_issues)} sample(s)")
            for line_no, col, byte, row in ascii_issues:
                msg.append(
                    f"    line {line_no}, col {col}, byte 0x{byte:02X} ({byte}): {_bytes_preview(row)}"
                )
        if svg_issues:
            msg.append(f"  non_svg_lines: {len(svg_issues)} sample(s)")
            for line_no, row in svg_issues:
                msg.append(f"    line {line_no}: {_bytes_preview(row)}")
        msg.extend(
            [
                "Fix with cleanup:",
                f"  python3 version/v7/scripts/prepare_ascii_dataset_v7.py --input {shlex.quote(str(dataset_path))} --output {shlex.quote(str(dataset_path))} --input-format text --ascii-mode xml_escape --svg-only",
            ]
        )
        raise SystemExit("\n".join(msg))

    return {
        "status": "pass",
        "path": str(dataset_path),
        "dataset_dir": str(dataset_path.parent),
        "dataset_name": dataset_path.name,
        "total_lines": int(len(rows)),
        "non_empty_lines": int(non_empty),
        "bytes": int(len(payload)),
        "require_ascii": bool(require_ascii),
        "require_svg_rows": bool(require_svg_rows),
        "ascii_violations": 0,
        "svg_violations": 0,
        "checks": {
            "ascii_gate": bool(require_ascii),
            "svg_row_gate": bool(require_svg_rows),
        },
    }


def _char_display(ch: str) -> str:
    if ch == "\n":
        return "\\n"
    if ch == "\r":
        return "\\r"
    if ch == "\t":
        return "\\t"
    cp = ord(ch)
    if cp < 32 or cp == 127:
        return f"\\x{cp:02X}"
    return ch


def _build_dataset_profile(dataset_path: Path, token_ids: list[int] | None = None, top_k: int = 16) -> dict[str, Any]:
    try:
        text = dataset_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = dataset_path.read_text(encoding="utf-8", errors="ignore")

    rows_all = text.splitlines()
    rows = [row for row in rows_all if row.strip()]
    line_lengths = [len(row) for row in rows]
    total_chars = len(text)

    bucket_edges = [32, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 4096]
    bucket_labels: list[str] = []
    lo = 0
    for hi in bucket_edges:
        bucket_labels.append(f"{lo}-{hi - 1}")
        lo = hi
    bucket_labels.append(f">={bucket_edges[-1]}")
    histogram = {label: 0 for label in bucket_labels}
    for n in line_lengths:
        idx = len(bucket_edges)
        for i, hi in enumerate(bucket_edges):
            if n < hi:
                idx = i
                break
        histogram[bucket_labels[idx]] += 1

    line_counts = Counter(rows)
    duplicate_items = [(line, count) for line, count in line_counts.items() if count > 1]
    duplicate_items.sort(key=lambda kv: (-kv[1], kv[0]))
    top_duplicates = [
        {"line": line[:240], "chars": int(len(line)), "count": int(count)}
        for line, count in duplicate_items[:top_k]
    ]
    duplicate_rows_total = int(sum(count for _, count in duplicate_items))

    char_counts = Counter(text)
    top_chars = [
        {"char": _char_display(ch), "codepoint": int(ord(ch)), "count": int(count)}
        for ch, count in char_counts.most_common(top_k)
    ]

    top_tokens: list[dict[str, Any]] = []
    if token_ids:
        token_counts = Counter(int(tok) for tok in token_ids)
        top_tokens = [
            {"id": int(tok), "count": int(count)}
            for tok, count in token_counts.most_common(top_k)
        ]

    avg_len = float(sum(line_lengths) / len(line_lengths)) if line_lengths else 0.0
    return {
        "path": str(dataset_path),
        "dataset_dir": str(dataset_path.parent),
        "dataset_name": dataset_path.name,
        "total_lines": int(len(rows_all)),
        "non_empty_lines": int(len(rows)),
        "total_chars": int(total_chars),
        "line_length": {
            "avg": avg_len,
            "min": int(min(line_lengths)) if line_lengths else 0,
            "max": int(max(line_lengths)) if line_lengths else 0,
            "histogram": histogram,
        },
        "duplicates": {
            "duplicate_unique_rows": int(len(duplicate_items)),
            "duplicate_rows_total": duplicate_rows_total,
            "top_rows": top_duplicates,
        },
        "top_chars": top_chars,
        "top_tokens": top_tokens,
    }


def _decode_with_ck_true_bpe(tokenizer_lib: Path, bin_dir: Path, ids: list[int]) -> str:
    with _TrueBPEHandle(tokenizer_lib, bin_dir) as handle:
        return handle.decode(ids)


def _collect_roundtrip_mismatches(expected: bytes, got: bytes, limit: int = 8) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    n = min(len(expected), len(got))
    for i in range(n):
        if expected[i] == got[i]:
            continue
        out.append(
            {
                "offset": int(i),
                "expected_byte": int(expected[i]),
                "got_byte": int(got[i]),
                "expected_char": chr(expected[i]) if 32 <= expected[i] <= 126 else ".",
                "got_char": chr(got[i]) if 32 <= got[i] <= 126 else ".",
            }
        )
        if len(out) >= limit:
            break
    if len(out) < limit and len(expected) != len(got):
        out.append(
            {
                "offset": int(n),
                "expected_len": int(len(expected)),
                "got_len": int(len(got)),
                "note": "length_mismatch",
            }
        )
    return out


def _build_roundtrip_report(
    tokenizer_mode: str,
    original_text: str,
    decoded_text: str,
    token_count: int,
) -> dict[str, Any]:
    expected = original_text.encode("utf-8")
    got = decoded_text.encode("utf-8")
    exact = expected == got
    min_len = min(len(expected), len(got))
    byte_matches = sum(1 for i in range(min_len) if expected[i] == got[i])
    byte_match_rate = float(byte_matches / max(1, max(len(expected), len(got))))

    exp_lines = original_text.splitlines()
    got_lines = decoded_text.splitlines()
    matched_lines = sum(1 for a, b in zip(exp_lines, got_lines) if a == b)
    line_match_rate = float(matched_lines / max(1, len(exp_lines)))

    return {
        "status": "pass" if exact else "fail",
        "tokenizer_mode": str(tokenizer_mode),
        "token_count": int(token_count),
        "input_bytes": int(len(expected)),
        "decoded_bytes": int(len(got)),
        "input_lines": int(len(exp_lines)),
        "decoded_lines": int(len(got_lines)),
        "exact_match": bool(exact),
        "byte_match_rate": byte_match_rate,
        "line_match_rate": line_match_rate,
        "mismatch_samples": [] if exact else _collect_roundtrip_mismatches(expected, got),
    }


def _truncate_text_preview(text: str, limit: int = 220) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "..."


def _evaluate_line_roundtrip(
    text: str,
    encode_line: Callable[[str], list[int]],
    decode_ids: Callable[[list[int]], str],
    line_no: int,
) -> dict[str, Any]:
    ids = encode_line(text)
    decoded = decode_ids(ids)
    exact = decoded == text
    return {
        "line_no": int(line_no),
        "token_count": int(len(ids)),
        "token_ids": [int(v) for v in ids[:96]],
        "token_ids_truncated": bool(len(ids) > 96),
        "source": _truncate_text_preview(text, limit=320),
        "decoded": _truncate_text_preview(decoded, limit=320),
        "exact_match": bool(exact),
    }


def _build_tokenizer_roundtrip_report(
    tokenizer_mode: str,
    dataset_path: Path,
    original_text: str,
    decoded_text: str,
    token_ids: list[int],
    encode_line: Callable[[str], list[int]],
    decode_ids: Callable[[list[int]], str],
    max_lines: int,
    sample_limit: int,
    tokenizer_json_path: str | None,
) -> dict[str, Any]:
    report = _build_roundtrip_report(
        tokenizer_mode=tokenizer_mode,
        original_text=original_text,
        decoded_text=decoded_text,
        token_count=len(token_ids),
    )

    line_results: list[dict[str, Any]] = []
    mismatch_rows: list[dict[str, Any]] = []
    exact_count = 0
    evaluated = 0
    non_empty_total = 0

    for line_no, row in enumerate(original_text.splitlines(), start=1):
        if not row.strip():
            continue
        non_empty_total += 1
        if evaluated >= max_lines:
            continue
        try:
            row_result = _evaluate_line_roundtrip(row, encode_line, decode_ids, line_no)
        except Exception as exc:
            row_result = {
                "line_no": int(line_no),
                "token_count": 0,
                "token_ids": [],
                "token_ids_truncated": False,
                "source": _truncate_text_preview(row, limit=320),
                "decoded": f"[roundtrip_error] {exc}",
                "exact_match": False,
                "error": str(exc),
            }
        evaluated += 1
        if bool(row_result.get("exact_match")):
            exact_count += 1
        else:
            if len(mismatch_rows) < max(1, int(sample_limit)):
                mismatch_rows.append(dict(row_result))
        if len(line_results) < max(1, int(sample_limit)):
            line_results.append(dict(row_result))

    report.update(
        {
            "dataset_path": str(dataset_path),
            "dataset_dir": str(dataset_path.parent),
            "dataset_name": dataset_path.name,
            "tokenizer_json_path": tokenizer_json_path,
            "line_eval": {
                "evaluated_lines": int(evaluated),
                "total_non_empty_lines": int(non_empty_total),
                "exact_match_lines": int(exact_count),
                "exact_match_rate": float(exact_count / max(1, evaluated)),
                "coverage_rate": float(evaluated / max(1, non_empty_total)),
                "max_lines": int(max_lines),
            },
            "sample_rows": line_results,
            "mismatch_rows": mismatch_rows,
        }
    )
    return report


def _extract_response_block(chat_stdout: str) -> str:
    marker = "Response:"
    if marker not in chat_stdout:
        return ""
    block = chat_stdout.split(marker, 1)[1]
    lower = block.lower()
    idx = lower.find("\nprompt eval:")
    if idx >= 0:
        block = block[:idx]
    return block.strip()


def _is_valid_svg_fragment(fragment: str) -> tuple[bool, str | None]:
    try:
        root = ET.fromstring(fragment)
    except Exception as exc:
        return False, str(exc)
    tag = root.tag
    if "}" in tag:
        tag = tag.split("}", 1)[1]
    if tag.lower() != "svg":
        return False, f"root_tag={tag}"
    return True, None


def _run_post_train_svg_eval(
    args: argparse.Namespace,
    run_dir: Path,
) -> tuple[dict[str, Any], str | None]:
    eval_prompt_arg = getattr(args, "eval_prompt", None)
    prompt = str(eval_prompt_arg or ("<svg" if args.require_svg_rows else "Hello"))
    eval_payload: dict[str, Any] = {
        "status": "skipped",
        "mode": "svg_output_eval",
        "prompt": prompt,
        "max_tokens": int(getattr(args, "eval_max_tokens", 160)),
        "temperature": float(getattr(args, "eval_temperature", 0.0)),
        "valid_svg_rate": 0.0,
        "closure_success_rate": 0.0,
        "repetition_loop_score": 1.0,
        "sample_count": 0,
        "valid_count": 0,
        "invalid_count": 0,
    }

    if not bool(getattr(args, "post_train_eval", True)):
        eval_payload["reason"] = "disabled_by_flag"
        return eval_payload, None
    if (not bool(getattr(args, "require_svg_rows", False))) and (eval_prompt_arg is None):
        eval_payload["reason"] = "skipped_non_svg_corpus"
        return eval_payload, None

    ck_build_dir = run_dir / ".ck_build"
    try:
        _run(
            [
                _python_exec(),
                str(CK_RUN),
                "run",
                str(run_dir),
                "--generate-only",
                "--context-len",
                str(args.context_len),
            ],
            cwd=ROOT,
        )
        result = _run_capture(
            [
                _python_exec(),
                str(CK_CHAT),
                "--model-dir",
                str(ck_build_dir),
                "--python-tokenizer",
                "--chat-template",
                "none",
                "--prompt",
                prompt,
                "--max-tokens",
                str(int(getattr(args, "eval_max_tokens", 160))),
                "--temperature",
                str(float(getattr(args, "eval_temperature", 0.0))),
            ],
            cwd=ROOT,
        )
    except Exception as exc:
        eval_payload["status"] = "error"
        eval_payload["reason"] = str(exc)
        return eval_payload, None

    response_text = _extract_response_block(result.stdout or "")
    open_tags = len(re.findall(r"<svg\b", response_text, flags=re.IGNORECASE))
    close_tags = len(re.findall(r"</svg>", response_text, flags=re.IGNORECASE))
    svg_fragments = re.findall(r"<svg\b.*?</svg>", response_text, flags=re.IGNORECASE | re.DOTALL)

    sample_cap = max(1, int(getattr(args, "eval_sample_limit", 12)))
    valid_count = 0
    sample_rows: list[dict[str, Any]] = []
    canonical_rows: list[str] = []
    for idx, frag in enumerate(svg_fragments):
        ok, err = _is_valid_svg_fragment(frag)
        canonical = " ".join(frag.split())
        canonical_rows.append(canonical)
        if ok:
            valid_count += 1
        if len(sample_rows) < sample_cap:
            sample_rows.append(
                {
                    "index": int(idx),
                    "valid": bool(ok),
                    "error": err,
                    "preview": _truncate_text_preview(canonical, limit=320),
                }
            )

    sample_count = len(svg_fragments)
    valid_rate = float(valid_count / max(1, sample_count))
    closure_rate = float(min(open_tags, close_tags) / max(1, open_tags))
    if canonical_rows:
        counts = Counter(canonical_rows)
        duplicate_rate = float(1.0 - (len(counts) / len(canonical_rows)))
        max_repeat_share = float(max(counts.values()) / len(canonical_rows))
    else:
        duplicate_rate = 1.0
        max_repeat_share = 1.0
    response_tokens = response_text.split()
    repeated_adjacent = 0
    for i in range(1, len(response_tokens)):
        if response_tokens[i] == response_tokens[i - 1]:
            repeated_adjacent += 1
    adjacent_repeat_rate = float(repeated_adjacent / max(1, len(response_tokens) - 1))
    loop_score = float(max(duplicate_rate, max_repeat_share, adjacent_repeat_rate))

    eval_payload.update(
        {
            "status": "ok",
            "response_chars": int(len(response_text)),
            "open_svg_tags": int(open_tags),
            "close_svg_tags": int(close_tags),
            "sample_count": int(sample_count),
            "valid_count": int(valid_count),
            "invalid_count": int(max(0, sample_count - valid_count)),
            "valid_svg_rate": valid_rate,
            "closure_success_rate": closure_rate,
            "repetition_loop_score": loop_score,
            "sample_rows": sample_rows,
            "response_preview": _truncate_text_preview(response_text, limit=1000),
        }
    )
    return eval_payload, str(ck_build_dir)


def _emit_data_lab_artifacts(
    run_dir: Path,
    dataset_qc: dict[str, Any],
    dataset_profile: dict[str, Any],
    tokenizer_roundtrip: dict[str, Any],
) -> dict[str, str]:
    run_dir.mkdir(parents=True, exist_ok=True)
    payloads = {
        "dataset_qc_json": run_dir / "dataset_qc.json",
        "dataset_profile_json": run_dir / "dataset_profile.json",
        "tokenizer_roundtrip_json": run_dir / "tokenizer_roundtrip.json",
    }
    docs = {
        "dataset_qc_json": dict(dataset_qc),
        "dataset_profile_json": dict(dataset_profile),
        "tokenizer_roundtrip_json": dict(tokenizer_roundtrip),
    }
    for key, path in payloads.items():
        doc = dict(docs[key])
        doc.setdefault("generated_at", datetime.now(timezone.utc).isoformat())
        _atomic_write_text(path, json.dumps(doc, indent=2))
    return {key: str(path) for key, path in payloads.items()}


def _make_corpus_dir_from_dataset(dataset_path: Path, work_dir: Path) -> Path:
    corpus_dir = work_dir / "corpus"
    corpus_dir.mkdir(parents=True, exist_ok=True)
    dst = corpus_dir / dataset_path.name
    try:
        raw = dataset_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        print(f"WARNING: {dataset_path} contains non-UTF-8 bytes; they will be dropped.", file=sys.stderr)
        raw = dataset_path.read_text(encoding="utf-8", errors="ignore")
    dst.write_text(raw, encoding="utf-8")
    return corpus_dir


def _sync_bpe_artifacts_to_run(run_dir: Path, tokenizer_json: Path, bpe_bin_dir: Path) -> Path:
    """Persist tokenizer artifacts in the run dir for inference-time reuse."""
    run_dir.mkdir(parents=True, exist_ok=True)
    dst_tok_json = run_dir / "tokenizer.json"
    shutil.copy2(tokenizer_json, dst_tok_json)

    dst_bin_dir = run_dir / "tokenizer_bin"
    dst_bin_dir.mkdir(parents=True, exist_ok=True)
    for name in ("tokenizer_meta.json", "vocab_offsets.bin", "vocab_strings.bin", "vocab_merges.bin"):
        src = bpe_bin_dir / name
        if not src.exists():
            raise RuntimeError(f"Missing BPE artifact for run sync: {src}")
        shutil.copy2(src, dst_bin_dir / name)
    return dst_bin_dir


def _load_ck_run_module():
    spec = importlib.util.spec_from_file_location("ck_run_v7_module_for_pipeline", CK_RUN)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module spec: {CK_RUN}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _resolve_run_adamw_hparams(run_dir: Path) -> dict[str, float]:
    defaults = {"beta1": 0.9, "beta2": 0.999, "eps": 1e-8, "weight_decay": 0.01}
    manifest = run_dir / "weights_manifest.json"
    if not manifest.exists():
        return defaults
    try:
        payload = _load_json(manifest)
    except Exception:
        return defaults
    cfg = payload.get("config") if isinstance(payload, dict) else None
    tr = cfg.get("training") if isinstance(cfg, dict) else None
    opt = tr.get("optimizer") if isinstance(tr, dict) else None
    adamw = opt.get("adamw") if isinstance(opt, dict) else None
    if not isinstance(adamw, dict):
        return defaults
    out = dict(defaults)
    for k in ("beta1", "beta2", "eps", "weight_decay"):
        v = adamw.get(k)
        if isinstance(v, (int, float)) and math.isfinite(float(v)):
            out[k] = float(v)
    return out


def _ensure_ck_runtime_for_cli(args: argparse.Namespace, run_dir: Path) -> None:
    module = _load_ck_run_module()
    ensure_fn = getattr(module, "_ensure_train_runtime_artifacts", None)
    if not callable(ensure_fn):
        raise RuntimeError("ck_run_v7.py missing _ensure_train_runtime_artifacts; cannot prepare runtime for ck-cli")
    adamw = _resolve_run_adamw_hparams(run_dir)
    runtime_defines: dict[str, Any] = {
        "CK_NUM_TOKENS": max(1, int(args.seq_len)),
        "CK_GRAD_ACCUM_STEPS": max(1, int(args.grad_accum)),
        "CK_TRAIN_USE_CE_PTREF": 0,
        "CK_MAX_GRAD_NORM": f"{float(args.max_grad_norm):.9g}",
        "CK_ADAMW_BETA1": f"{float(adamw['beta1']):.9g}",
        "CK_ADAMW_BETA2": f"{float(adamw['beta2']):.9g}",
        "CK_ADAMW_EPS": f"{float(adamw['eps']):.9g}",
        "CK_ADAMW_WEIGHT_DECAY": f"{float(adamw['weight_decay']):.9g}",
    }
    ensure_fn(
        run_dir=run_dir,
        python_exec=_python_exec(),
        strict=False,
        runtime_defines=runtime_defines,
        train_tokens=max(1, int(args.seq_len)),
        extra_cflags=None,
    )


def _sha256_file(path: Path) -> str:
    """Return hex SHA-256 of a file (streaming, constant memory)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _build_training_pipeline_payload(
    args: argparse.Namespace,
    run_dir: Path,
    dataset_path: Path,
    bpe_artifacts: dict[str, Any],
    ck_loss: dict[str, Any],
    dataset_qc: dict[str, Any] | None = None,
    dataset_profile: dict[str, Any] | None = None,
    tokenizer_roundtrip: dict[str, Any] | None = None,
    data_lab_artifacts: dict[str, str] | None = None,
    post_train_eval: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build ``training_pipeline_latest.json`` in the schema the visualizer expects.

    Schema: ``ck.training_pipeline.v1``  (see ck_run_v7.py _build_training_pipeline_payload).
    """
    now_iso = datetime.now(timezone.utc).isoformat()
    active_stage = "pretrain"
    stage_names = ["pretrain", "sft", "dpo", "grpo", "ppo"]
    stage_timeline = [
        {
            "stage": s,
            "order": i,
            "status": "active" if s == active_stage else ("planned" if i > 0 else "completed"),
            "active": s == active_stage,
        }
        for i, s in enumerate(stage_names)
    ]

    # ── data provenance ─────────────────────────────────────────
    dataset_hash = _sha256_file(dataset_path) if dataset_path.exists() else None
    dataset_size = dataset_path.stat().st_size if dataset_path.exists() else 0
    token_count = bpe_artifacts.get("token_count") or 0
    if not token_count and dataset_path.exists():
        # byte tokenizer: one token per byte
        token_count = dataset_size

    data_provenance = [
        {
            "stage": active_stage,
            "dataset_name": dataset_path.name,
            "source_path": str(dataset_path),
            "split": "train",
            "token_count": int(token_count),
            "byte_size": int(dataset_size),
            "hash": {"algorithm": "sha256", "value": dataset_hash} if dataset_hash else None,
            "sampling": {},
            "packing": {},
        }
    ]

    # ── tokenizer lineage ───────────────────────────────────────
    tokenizer_kind = str(args.tokenizer)
    vocab_size = _read_run_vocab_size(run_dir) or (
        int(args.bpe_vocab_size) if _is_bpe_tokenizer_mode(tokenizer_kind) else 256
    )
    tokenizer_lineage: dict[str, Any] = {
        "type": tokenizer_kind,
        "vocab_size": int(vocab_size),
        "template": str(getattr(args, "template", "qwen3")),
    }
    if _is_bpe_tokenizer_mode(tokenizer_kind):
        tok_json_path = bpe_artifacts.get("tokenizer_json")
        if tok_json_path:
            tokenizer_lineage["tokenizer_path"] = str(tok_json_path)
            tok_path = Path(tok_json_path)
            if tok_path.exists():
                tokenizer_lineage["tokenizer_sha256"] = _sha256_file(tok_path)
        tokenizer_lineage["bpe_vocab_size"] = int(args.bpe_vocab_size)
        tokenizer_lineage["bpe_min_freq"] = int(args.bpe_min_freq)
        tokenizer_lineage["bpe_mode"] = "ascii_bpe" if tokenizer_kind == "ascii_bpe" else "bytelevel_bpe"

    data_lab = {
        "dataset_path": str(dataset_path),
        "dataset_dir": str(dataset_path.parent),
        "tokenizer_json_path": (
            bpe_artifacts.get("run_tokenizer_json")
            or bpe_artifacts.get("tokenizer_json")
            or tokenizer_lineage.get("tokenizer_path")
        ),
        "artifacts": dict(data_lab_artifacts or {}),
        "dataset_qc": dict(dataset_qc or {}),
        "dataset_profile": dict(dataset_profile or {}),
        "tokenizer_roundtrip": dict(tokenizer_roundtrip or {}),
    }
    if isinstance(post_train_eval, dict) and post_train_eval:
        data_lab["post_train_eval"] = dict(post_train_eval)

    # ── execution ───────────────────────────────────────────────
    steps = ck_loss.get("steps", 0) if isinstance(ck_loss, dict) else 0
    tokens_per_update = int(args.seq_len) * int(args.grad_accum)

    # ── model dims from manifest ────────────────────────────────
    train_dims: dict[str, Any] = {}
    manifest_path = run_dir / "weights_manifest.json"
    if manifest_path.exists():
        try:
            manifest = _load_json(manifest_path)
            cfg = manifest.get("config") if isinstance(manifest, dict) else {}
            if isinstance(cfg, dict):
                for k in ("vocab_size", "embed_dim", "hidden_dim", "num_layers",
                           "num_heads", "num_kv_heads", "head_dim", "context_length"):
                    v = cfg.get(k)
                    if v is not None:
                        train_dims[k] = v
        except Exception:
            pass

    return {
        "schema": "ck.training_pipeline.v1",
        "generated_at": now_iso,
        "active_stage": active_stage,
        "stage_timeline": stage_timeline,
        "backend": "ck",
        "optimizer": {
            "name": "adamw",
            "lr": float(args.lr),
            "hparams": {
                "max_grad_norm": float(args.max_grad_norm),
                "seed": int(args.seed),
            },
        },
        "execution": {
            "epochs": int(args.epochs),
            "steps": int(steps),
            "micro_steps": 0,
            "optimizer_steps": 0,
            "seq_len": int(args.seq_len),
            "grad_accum": int(args.grad_accum),
            "tokens_total": int(args.total_tokens),
            "tokens_per_update": int(tokens_per_update),
            "processed_tokens": int(steps) * int(args.seq_len) if steps else 0,
        },
        "train_dims": train_dims,
        "data_provenance": data_provenance,
        "tokenizer_lineage": tokenizer_lineage,
        "data_lab": data_lab,
        "sources": {
            "summary": "train_data_pipeline_v7",
            "run_dir": str(run_dir),
        },
    }


def _run_ck_train(
    args: argparse.Namespace,
    dataset_path: Path,
    token_file: Path | None,
    ck_json: Path,
) -> None:
    run_dir = Path(args.run).expanduser().resolve()
    train_driver = str(getattr(args, "train_driver", "ck_run") or "ck_run").strip().lower()
    if train_driver == "ck_cli":
        if token_file is None:
            raise RuntimeError("ck_cli train driver requires --tokenizer bpe/ascii_bpe or prebuilt --token-file-out.")
        _ensure_binary(CK_CLI_BIN, "ck-cli-v7")
        _ensure_ck_runtime_for_cli(args, run_dir)
        cmd = [
            str(CK_CLI_BIN),
            "train",
            "--run",
            str(run_dir),
            "--train-token-file",
            str(token_file),
            "--train-json-out",
            str(ck_json),
            "--train-epochs",
            str(args.epochs),
            "--train-seq-len",
            str(args.seq_len),
            "--train-total-tokens",
            str(args.total_tokens),
            "--train-grad-accum",
            str(args.grad_accum),
            "--train-lr",
            str(args.lr),
        ]
        log_every = int(getattr(args, "ck_cli_log_every", 0) or 0)
        if log_every > 0:
            cmd.extend(["--log-every", str(log_every)])
        if bool(getattr(args, "verbose", False)):
            cmd.append("--verbose")
        _run(cmd, cwd=ROOT)
        return

    py = _python_exec()
    cmd = [
        py,
        str(CK_RUN),
        "train",
        "--run",
        str(run_dir),
        "--backend",
        "ck",
        "--train-epochs",
        str(args.epochs),
        "--train-seq-len",
        str(args.seq_len),
        "--train-total-tokens",
        str(args.total_tokens),
        "--train-grad-accum",
        str(args.grad_accum),
        "--train-lr",
        str(args.lr),
        "--train-max-grad-norm",
        str(args.max_grad_norm),
        "--train-seed",
        str(args.seed),
        "--train-json-out",
        str(ck_json),
    ]
    if args.enforce_production_safety:
        cmd.append("--enforce-production-safety")
    if token_file is not None:
        cmd.extend(["--train-token-file", str(token_file)])
    else:
        cmd.extend(["--data", str(dataset_path)])
    _run(cmd, cwd=ROOT)


def _run_torch_ref(
    args: argparse.Namespace,
    dataset_path: Path,
    torch_json: Path,
    token_file: Path | None = None,
) -> None:
    py = _python_exec()
    cmd = [
        py,
        str(TORCH_REF),
        "--run-dir",
        str(Path(args.run).expanduser().resolve()),
        "--epochs",
        str(args.epochs),
        "--seq-len",
        str(args.seq_len),
        "--total-tokens",
        str(args.total_tokens),
        "--lr",
        str(args.lr),
        "--max-grad-norm",
        str(args.max_grad_norm),
        "--seed",
        str(args.seed),
        "--json-out",
        str(torch_json),
    ]
    if token_file is not None:
        cmd.extend(["--token-file", str(token_file)])
    else:
        cmd.extend(["--data", str(dataset_path)])
    _run(cmd, cwd=ROOT)


def main() -> int:
    ap = argparse.ArgumentParser(description="High-level v7 dataset/tokenizer/train pipeline")
    ap.add_argument("--run", required=True, help="Existing v7 run-dir (created by ck_run_v7.py init)")
    ap.add_argument("--init-if-missing", action="store_true", help="Auto-run v7 init when --run does not exist")
    ap.add_argument("--init", default="xavier_uniform", choices=["normal_0p02", "xavier_uniform", "xavier_normal", "kaiming_uniform", "zeros"])
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--vocab-size", type=int, default=None, help="Run vocab size for init (default: 256 byte, bpe-vocab-size for bpe/ascii_bpe)")
    ap.add_argument("--embed-dim", type=int, default=128)
    ap.add_argument("--hidden-dim", type=int, default=256)
    ap.add_argument("--num-heads", type=int, default=8)
    ap.add_argument("--num-kv-heads", type=int, default=4)
    ap.add_argument("--context-len", type=int, default=128)
    ap.add_argument("--template", default="qwen3")
    ap.add_argument("--data", default=None, help="UTF-8 training text file path")
    ap.add_argument("--dataset-repeats", type=int, default=10, help="If --data missing, create repeated SVG rows")
    ap.add_argument("--tokenizer", choices=["byte", "bpe", "ascii_bpe"], default="byte", help="Tokenization path for training")
    ap.set_defaults(require_ascii_data=None)
    ap.add_argument(
        "--require-ascii-data",
        dest="require_ascii_data",
        action="store_true",
        help="Fail if dataset contains non-ASCII bytes (default: enabled for --tokenizer ascii_bpe)",
    )
    ap.add_argument(
        "--no-require-ascii-data",
        dest="require_ascii_data",
        action="store_false",
        help="Allow non-ASCII dataset bytes even with --tokenizer ascii_bpe",
    )
    ap.add_argument(
        "--require-svg-rows",
        action="store_true",
        help="Fail if any non-empty dataset row does not start with <svg",
    )
    ap.add_argument("--work-dir", default=None, help="Optional work dir for generated artifacts")
    # Training hyper-parameter defaults.  These are always passed explicitly to
    # child scripts (ck_run_v7.py, train_qwen3_torch_from_run_v7.py), so child
    # defaults do NOT matter when invoked through this pipeline.  If you change
    # a default here, the new value propagates automatically.
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--seq-len", type=int, default=32)
    ap.add_argument("--total-tokens", type=int, default=1024)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--enforce-production-safety", action="store_true")
    ap.add_argument("--with-torch-ref", action="store_true", help="Run torch ref too (byte/bpe/ascii_bpe via token-file)")
    ap.set_defaults(open_visualizer=True)
    ap.add_argument("--open-visualizer", dest="open_visualizer", action="store_true",
                    help="Generate v7 IR visualizer HTML after training (default: enabled)")
    ap.add_argument("--no-open-visualizer", dest="open_visualizer", action="store_false",
                    help="Skip v7 IR visualizer HTML generation")
    ap.add_argument("--json-out", default=None, help="Optional pipeline report JSON")
    ap.add_argument("--bpe-vocab-size", type=int, default=1024)
    ap.add_argument("--bpe-min-freq", type=int, default=2)
    ap.add_argument("--bpe-threads", type=int, default=4)
    ap.add_argument(
        "--train-driver",
        choices=["ck_run", "ck_cli"],
        default="ck_run",
        help="Training executor (ck_run=python ctypes runtime, ck_cli=native C CLI runtime)",
    )
    ap.add_argument(
        "--ck-cli-log-every",
        type=int,
        default=0,
        help="When --train-driver ck_cli, print progress every N steps (0=auto cadence)",
    )
    ap.add_argument(
        "--token-file-out",
        default=None,
        help="Optional canonical path to write final train token stream",
    )
    ap.add_argument(
        "--roundtrip-max-lines",
        type=int,
        default=512,
        help="Max non-empty rows to evaluate for tokenizer line roundtrip metrics",
    )
    ap.add_argument(
        "--roundtrip-sample-limit",
        type=int,
        default=16,
        help="Number of line samples/mismatch samples to persist in tokenizer_roundtrip.json",
    )
    ap.add_argument(
        "--profile-top-k",
        type=int,
        default=16,
        help="Top-K entries for dataset profile frequency tables",
    )
    ap.add_argument(
        "--strict-data-gates",
        action="store_true",
        help="Fail pipeline when strict data-quality gates are violated",
    )
    ap.add_argument(
        "--min-valid-svg-rate",
        type=float,
        default=0.70,
        help="Strict gate threshold for post-train valid SVG rate (0..1)",
    )
    ap.set_defaults(post_train_eval=True)
    ap.add_argument(
        "--post-train-eval",
        dest="post_train_eval",
        action="store_true",
        help="Run post-train SVG output quality eval and write post_train_eval.json",
    )
    ap.add_argument(
        "--no-post-train-eval",
        dest="post_train_eval",
        action="store_false",
        help="Skip post-train output quality eval",
    )
    ap.add_argument(
        "--eval-prompt",
        default=None,
        help="Prompt used for post-train eval (default: <svg when --require-svg-rows else Hello)",
    )
    ap.add_argument(
        "--eval-max-tokens",
        type=int,
        default=160,
        help="Max generated tokens for post-train eval",
    )
    ap.add_argument(
        "--eval-temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for post-train eval",
    )
    ap.add_argument(
        "--eval-sample-limit",
        type=int,
        default=12,
        help="How many sampled generated SVG rows to keep in post_train_eval.json",
    )
    ap.add_argument(
        "--prepare-only",
        action="store_true",
        help="Prepare dataset + tokenizer + token stream and stop before training",
    )
    ap.add_argument("--verbose", action="store_true", help="Verbose pipeline logs")
    args = ap.parse_args()

    # ── Validate numeric arguments ──────────────────────────────
    _errors: list[str] = []
    if args.epochs < 1:
        _errors.append(f"--epochs must be >= 1, got {args.epochs}")
    if args.seq_len < 1:
        _errors.append(f"--seq-len must be >= 1, got {args.seq_len}")
    if args.total_tokens < args.seq_len + 1:
        _errors.append(
            f"--total-tokens ({args.total_tokens}) must be >= --seq-len + 1 ({args.seq_len + 1})"
        )
    if args.grad_accum < 1:
        _errors.append(f"--grad-accum must be >= 1, got {args.grad_accum}")
    if args.lr <= 0:
        _errors.append(f"--lr must be > 0, got {args.lr}")
    if args.max_grad_norm <= 0:
        _errors.append(f"--max-grad-norm must be > 0, got {args.max_grad_norm}")
    if args.layers < 1:
        _errors.append(f"--layers must be >= 1, got {args.layers}")
    if args.embed_dim < 1:
        _errors.append(f"--embed-dim must be >= 1, got {args.embed_dim}")
    if args.hidden_dim < 1:
        _errors.append(f"--hidden-dim must be >= 1, got {args.hidden_dim}")
    if args.bpe_vocab_size < 2:
        _errors.append(f"--bpe-vocab-size must be >= 2, got {args.bpe_vocab_size}")
    if args.roundtrip_max_lines < 1:
        _errors.append(f"--roundtrip-max-lines must be >= 1, got {args.roundtrip_max_lines}")
    if args.roundtrip_sample_limit < 1:
        _errors.append(f"--roundtrip-sample-limit must be >= 1, got {args.roundtrip_sample_limit}")
    if args.profile_top_k < 1:
        _errors.append(f"--profile-top-k must be >= 1, got {args.profile_top_k}")
    if not (0.0 <= float(args.min_valid_svg_rate) <= 1.0):
        _errors.append(f"--min-valid-svg-rate must be in [0,1], got {args.min_valid_svg_rate}")
    if args.eval_max_tokens < 1:
        _errors.append(f"--eval-max-tokens must be >= 1, got {args.eval_max_tokens}")
    if args.eval_sample_limit < 1:
        _errors.append(f"--eval-sample-limit must be >= 1, got {args.eval_sample_limit}")
    if args.eval_temperature < 0.0:
        _errors.append(f"--eval-temperature must be >= 0, got {args.eval_temperature}")
    if _errors:
        raise SystemExit("ERROR: invalid arguments:\\n  " + "\\n  ".join(_errors))

    if args.require_ascii_data is None:
        args.require_ascii_data = args.tokenizer == "ascii_bpe"

    run_dir = Path(args.run).expanduser().resolve()
    if not run_dir.exists():
        if not args.init_if_missing:
            raise SystemExit(
                f"ERROR: run-dir not found: {run_dir}\n"
                "Hint: pass --init-if-missing to bootstrap automatically."
            )
        init_vocab_size = int(args.vocab_size) if args.vocab_size is not None else (
            int(args.bpe_vocab_size) if _is_bpe_tokenizer_mode(args.tokenizer) else 256
        )
        _run(
            [
                _python_exec(),
                str(CK_RUN),
                "init",
                "--run",
                str(run_dir),
                "--init",
                str(args.init),
                "--layers",
                str(args.layers),
                "--vocab-size",
                str(init_vocab_size),
                "--embed-dim",
                str(args.embed_dim),
                "--hidden-dim",
                str(args.hidden_dim),
                "--num-heads",
                str(args.num_heads),
                "--num-kv-heads",
                str(args.num_kv_heads),
                "--context-len",
                str(args.context_len),
                "--template",
                str(args.template),
                "--train-seed",
                str(args.seed),
            ],
            cwd=ROOT,
        )

    if args.work_dir:
        work_dir = Path(args.work_dir).expanduser().resolve()
        work_dir.mkdir(parents=True, exist_ok=True)
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        work_dir = run_dir / ".ck_pipeline" / f"{args.tokenizer}_{stamp}"
        work_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = Path(args.data).expanduser().resolve() if args.data else (work_dir / "svg_train.txt")
    if args.data:
        if not dataset_path.exists():
            raise SystemExit(f"ERROR: training data file not found: {dataset_path}")
    else:
        _write_svg_dataset(dataset_path, args.dataset_repeats)

    dataset_qc = _validate_dataset_rows(
        dataset_path,
        require_ascii=bool(args.require_ascii_data),
        require_svg_rows=bool(args.require_svg_rows),
    )
    print(
        "[dataset-qc] "
        f"lines={dataset_qc['non_empty_lines']}/{dataset_qc['total_lines']} "
        f"bytes={dataset_qc['bytes']} "
        f"require_ascii={dataset_qc['require_ascii']} "
        f"require_svg_rows={dataset_qc['require_svg_rows']}"
    )

    ck_json = work_dir / "train_ck.json"
    torch_json = work_dir / "train_torch_ref.json"
    token_file: Path | None = None
    bpe_artifacts: dict[str, Any] = {}
    tokenizer_json_for_roundtrip: str | None = None
    token_ids_all: list[int] = []
    decoded_text: str = ""
    dataset_text = dataset_path.read_text(encoding="utf-8", errors="ignore")
    encode_line_fn: Callable[[str], list[int]]
    decode_ids_fn: Callable[[list[int]], str]
    bpe_handle: _TrueBPEHandle | None = None

    if _is_bpe_tokenizer_mode(args.tokenizer):
        _ensure_binary(BPE_BIN, "ck-bpe-train")
        _ensure_binary(TOKENIZER_LIB, "tokenizer")
        corpus_dir = _make_corpus_dir_from_dataset(dataset_path, work_dir)
        tokenizer_json = work_dir / "tokenizer.json"
        bpe_bin_dir = work_dir / "bpe_bin"
        bpe_bin_dir.mkdir(parents=True, exist_ok=True)
        bpe_cmd = [
            str(BPE_BIN),
            "--corpus-dir",
            str(corpus_dir),
            "--out",
            str(tokenizer_json),
            "--binary-out-dir",
            str(bpe_bin_dir),
            "--vocab-size",
            str(args.bpe_vocab_size),
            "--min-freq",
            str(args.bpe_min_freq),
            "--threads",
            str(args.bpe_threads),
        ]
        if args.tokenizer == "ascii_bpe":
            bpe_cmd.append("--ascii-only")
        _run(bpe_cmd, cwd=ROOT)
        bpe_handle = _TrueBPEHandle(TOKENIZER_LIB, bpe_bin_dir)
        ids = _encode_large_text_with_bpe_handle(bpe_handle, dataset_text)
        if len(ids) <= 1:
            raise SystemExit("ERROR: BPE encoding produced <=1 token; provide richer data.")
        run_bpe_bin_dir = _sync_bpe_artifacts_to_run(run_dir, tokenizer_json, bpe_bin_dir)
        run_vocab = _read_run_vocab_size(run_dir)
        if isinstance(run_vocab, int) and run_vocab > 0:
            max_id = int(max(ids))
            if max_id >= run_vocab:
                raise SystemExit(
                    "ERROR: BPE token ids exceed run vocab size.\n"
                    f"  run vocab_size: {run_vocab}\n"
                    f"  max token id:   {max_id}\n"
                    "Fix: re-init run-dir with --vocab-size >= --bpe-vocab-size (or >= max token id + 1)."
                )
        token_file = work_dir / "train_tokens.txt"
        _atomic_write_text(token_file, "\n".join(str(v) for v in ids) + "\n")
        token_ids_all = [int(v) for v in ids]
        tokenizer_json_for_roundtrip = str(run_dir / "tokenizer.json")
        decoded_text = bpe_handle.decode(token_ids_all)
        encode_line_fn = bpe_handle.encode
        decode_ids_fn = bpe_handle.decode
        bpe_artifacts = {
            "tokenizer_json": str(tokenizer_json),
            "binary_dir": str(bpe_bin_dir),
            "run_tokenizer_json": str(run_dir / "tokenizer.json"),
            "run_binary_dir": str(run_bpe_bin_dir),
            "token_file": str(token_file),
            "token_count": int(len(ids)),
            "mode": "ascii_bpe" if args.tokenizer == "ascii_bpe" else "bytelevel_bpe",
        }

    if str(args.train_driver) == "ck_cli" and token_file is None:
        # Native ck-cli train path consumes deterministic integer token streams.
        ids = list(dataset_path.read_bytes())
        if len(ids) <= 1:
            raise SystemExit("ERROR: byte tokenizer path produced <=1 token; provide richer data.")
        token_file = work_dir / "train_tokens.txt"
        _atomic_write_text(token_file, "\n".join(str(v) for v in ids) + "\n")

    if not _is_bpe_tokenizer_mode(args.tokenizer):
        raw = dataset_path.read_bytes()
        token_ids_all = [int(v) for v in raw]
        decoded_text = raw.decode("utf-8", errors="replace")
        encode_line_fn = lambda row: [int(v) for v in row.encode("utf-8")]
        decode_ids_fn = lambda row_ids: bytes(int(v) & 0xFF for v in row_ids).decode("utf-8", errors="replace")

    if args.token_file_out and token_file is not None:
        token_file_out = Path(args.token_file_out).expanduser().resolve()
        token_file_out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(token_file, token_file_out)
        token_file = token_file_out
        if bpe_artifacts:
            bpe_artifacts["token_file"] = str(token_file)

    dataset_profile = _build_dataset_profile(
        dataset_path,
        token_ids=token_ids_all,
        top_k=max(1, int(args.profile_top_k)),
    )
    tokenizer_roundtrip = _build_tokenizer_roundtrip_report(
        tokenizer_mode=str(args.tokenizer),
        dataset_path=dataset_path,
        original_text=dataset_text,
        decoded_text=decoded_text,
        token_ids=token_ids_all,
        encode_line=encode_line_fn,
        decode_ids=decode_ids_fn,
        max_lines=max(1, int(args.roundtrip_max_lines)),
        sample_limit=max(1, int(args.roundtrip_sample_limit)),
        tokenizer_json_path=tokenizer_json_for_roundtrip,
    )
    data_lab_artifacts = _emit_data_lab_artifacts(
        run_dir=run_dir,
        dataset_qc=dataset_qc,
        dataset_profile=dataset_profile,
        tokenizer_roundtrip=tokenizer_roundtrip,
    )
    if bpe_handle is not None:
        bpe_handle.close()
    if args.strict_data_gates and args.tokenizer == "ascii_bpe":
        if not bool(tokenizer_roundtrip.get("exact_match")):
            raise SystemExit(
                "ERROR: strict data gate failed (ascii_bpe roundtrip).\n"
                f"  exact_match: {tokenizer_roundtrip.get('exact_match')}\n"
                f"  artifact:    {data_lab_artifacts.get('tokenizer_roundtrip_json')}"
            )

    if args.prepare_only:
        report = {
            "format": "v7-train-data-pipeline",
            "run_dir": str(run_dir),
            "dataset": str(dataset_path),
            "tokenizer": str(args.tokenizer),
            "train_driver": str(args.train_driver),
            "prepare_only": True,
            "artifacts": {
                "work_dir": str(work_dir),
                "token_file": str(token_file) if token_file is not None else None,
                "bpe": bpe_artifacts or None,
                "data_lab": data_lab_artifacts,
            },
            "dataset_qc": dataset_qc,
            "dataset_profile": dataset_profile,
            "tokenizer_roundtrip": tokenizer_roundtrip,
        }
        if args.json_out:
            out_path = Path(args.json_out).expanduser().resolve()
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

        prepare_pipeline = _build_training_pipeline_payload(
            args,
            run_dir,
            dataset_path,
            bpe_artifacts,
            ck_loss={},
            dataset_qc=dataset_qc,
            dataset_profile=dataset_profile,
            tokenizer_roundtrip=tokenizer_roundtrip,
            data_lab_artifacts=data_lab_artifacts,
            post_train_eval={"status": "skipped", "reason": "prepare_only"},
        )
        _atomic_write_text(run_dir / "training_pipeline_latest.json", json.dumps(prepare_pipeline, indent=2))
        print("v7 train pipeline prepared")
        print(f"  run_dir:   {run_dir}")
        print(f"  dataset:   {dataset_path}")
        print(f"  tokenizer: {args.tokenizer}")
        print(f"  driver:    {args.train_driver}")
        if token_file is not None:
            print(f"  token_file:{token_file}")
        print(
            "  roundtrip:"
            f" exact={tokenizer_roundtrip.get('exact_match')}"
            f" line_rate={tokenizer_roundtrip.get('line_eval', {}).get('exact_match_rate', 0.0):.4f}"
        )
        print(f"  data_lab:  {data_lab_artifacts.get('dataset_profile_json')}")
        return 0

    _run_ck_train(args, dataset_path, token_file, ck_json)

    if args.with_torch_ref:
        _run_torch_ref(args, dataset_path, torch_json, token_file=token_file)

    post_train_eval, eval_model_dir = _run_post_train_svg_eval(args, run_dir)
    post_train_eval_path: str | None = None
    if isinstance(post_train_eval, dict) and post_train_eval:
        post_eval_doc = dict(post_train_eval)
        if eval_model_dir:
            post_eval_doc["model_dir"] = str(eval_model_dir)
        post_eval_doc.setdefault("generated_at", datetime.now(timezone.utc).isoformat())
        post_eval_file = run_dir / "post_train_eval.json"
        _atomic_write_text(post_eval_file, json.dumps(post_eval_doc, indent=2))
        post_train_eval_path = str(post_eval_file)

    if args.strict_data_gates and bool(args.require_svg_rows):
        if post_train_eval.get("status") != "ok":
            raise SystemExit(
                "ERROR: strict data gate failed (post-train eval unavailable).\n"
                f"  status: {post_train_eval.get('status')}\n"
                f"  reason: {post_train_eval.get('reason')}"
            )
        valid_svg_rate = float(post_train_eval.get("valid_svg_rate", 0.0))
        if valid_svg_rate < float(args.min_valid_svg_rate):
            raise SystemExit(
                "ERROR: strict data gate failed (valid SVG rate).\n"
                f"  valid_svg_rate: {valid_svg_rate:.4f}\n"
                f"  threshold:      {float(args.min_valid_svg_rate):.4f}\n"
                f"  artifact:       {post_train_eval_path or (run_dir / 'post_train_eval.json')}"
            )

    report = {
        "format": "v7-train-data-pipeline",
        "run_dir": str(run_dir),
        "dataset": str(dataset_path),
        "tokenizer": str(args.tokenizer),
        "train_driver": str(args.train_driver),
        "training": {
            "epochs": int(args.epochs),
            "seq_len": int(args.seq_len),
            "total_tokens": int(args.total_tokens),
            "grad_accum": int(args.grad_accum),
            "lr": float(args.lr),
            "max_grad_norm": float(args.max_grad_norm),
            "seed": int(args.seed),
        },
        "artifacts": {
            "work_dir": str(work_dir),
            "ck_json": str(ck_json),
            "torch_json": str(torch_json) if torch_json.exists() else None,
            "token_file": str(token_file) if token_file is not None else None,
            "bpe": bpe_artifacts or None,
            "data_lab": data_lab_artifacts,
            "post_train_eval_json": post_train_eval_path,
        },
        "dataset_qc": dataset_qc,
        "dataset_profile": dataset_profile,
        "tokenizer_roundtrip": tokenizer_roundtrip,
        "post_train_eval": post_train_eval,
        "ck_loss": {},
        "torch_loss": {},
    }

    if ck_json.exists():
        try:
            report["ck_loss"] = _loss_stats(_load_json(ck_json))
        except (json.JSONDecodeError, OSError) as e:
            print(f"WARNING: could not read CK JSON output ({ck_json}): {e}", file=sys.stderr)
    if torch_json.exists():
        try:
            report["torch_loss"] = _loss_stats(_load_json(torch_json))
        except (json.JSONDecodeError, OSError) as e:
            print(f"WARNING: could not read torch JSON output ({torch_json}): {e}", file=sys.stderr)

    out_path = Path(args.json_out).expanduser().resolve() if args.json_out else (work_dir / "pipeline_report.json")
    _atomic_write_text(out_path, json.dumps(report, indent=2))

    # ── Emit training_pipeline_latest.json for the IR visualizer ──
    training_pipeline = _build_training_pipeline_payload(
        args, run_dir, dataset_path, bpe_artifacts,
        ck_loss=report.get("ck_loss", {}),
        dataset_qc=dataset_qc,
        dataset_profile=dataset_profile,
        tokenizer_roundtrip=tokenizer_roundtrip,
        data_lab_artifacts={
            **dict(data_lab_artifacts),
            **({"post_train_eval_json": post_train_eval_path} if post_train_eval_path else {}),
        },
        post_train_eval=post_train_eval,
    )
    pipeline_json_path = run_dir / "training_pipeline_latest.json"
    _atomic_write_text(pipeline_json_path, json.dumps(training_pipeline, indent=2))

    print("v7 train pipeline complete")
    print(f"  run_dir:   {run_dir}")
    print(f"  dataset:   {dataset_path}")
    print(f"  tokenizer: {args.tokenizer}")
    print(f"  driver:    {args.train_driver}")
    print(f"  report:    {out_path}")
    print(
        "  roundtrip:"
        f" exact={tokenizer_roundtrip.get('exact_match')}"
        f" line_rate={tokenizer_roundtrip.get('line_eval', {}).get('exact_match_rate', 0.0):.4f}"
    )
    if post_train_eval_path:
        print(
            "  post_eval:"
            f" valid_svg_rate={float(post_train_eval.get('valid_svg_rate', 0.0)):.4f}"
            f" closure_rate={float(post_train_eval.get('closure_success_rate', 0.0)):.4f}"
            f" loop_score={float(post_train_eval.get('repetition_loop_score', 1.0)):.4f}"
        )
        print(f"  post_eval_json: {post_train_eval_path}")
    if report.get("ck_loss"):
        ck = report["ck_loss"]
        if isinstance(ck, dict) and ck.get("steps", 0):
            print(
                "  CK loss:   "
                f"first={ck.get('first'):.6f} final={ck.get('final'):.6f} "
                f"min={ck.get('min'):.6f} (step={ck.get('min_step')})"
            )
    if report.get("torch_loss"):
        pt = report["torch_loss"]
        if isinstance(pt, dict) and pt.get("steps", 0):
            print(
                "  PT loss:   "
                f"first={pt.get('first'):.6f} final={pt.get('final'):.6f} "
                f"min={pt.get('min'):.6f} (step={pt.get('min_step')})"
            )

    if args.open_visualizer:
        _run(
            [
                _python_exec(),
                str(OPEN_VIS),
                "--generate",
                "--run",
                str(run_dir),
                "--html-only",
            ],
            cwd=ROOT,
        )
        print("  visualizer: generated via open_ir_visualizer.py --generate --run ... --html-only")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

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
PROMOTE_CKPT = ROOT / "version" / "v7" / "scripts" / "promote_latest_checkpoint_v7.py"
PACK_TOKENS = ROOT / "version" / "v7" / "scripts" / "pack_training_tokens_v7.py"
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


def _promote_checkpoint(
    run_dir: Path,
    *,
    strict: bool,
    step: int | None = None,
    purpose: str = "eval",
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "status": "skipped",
        "reason": f"checkpoint_promotion_not_attempted:{purpose}",
    }
    if not PROMOTE_CKPT.exists():
        payload = {"status": "skipped", "reason": f"missing_script:{PROMOTE_CKPT}"}
        if strict:
            raise SystemExit(
                "ERROR: strict mode requires checkpoint promotion.\n"
                f"  missing: {PROMOTE_CKPT}"
            )
        return payload
    try:
        cmd = [
            _python_exec(),
            str(PROMOTE_CKPT),
            "--run",
            str(run_dir),
        ]
        if step is not None:
            cmd.extend(["--step", str(int(step))])
        _run(
            cmd,
            cwd=ROOT,
        )
        return {
            "status": "ok",
            "strategy": "exact_step" if step is not None else "latest_checkpoint",
            "step": int(step) if step is not None else None,
            "purpose": str(purpose),
        }
    except Exception as exc:
        payload = {"status": "error", "reason": str(exc)}
        if strict:
            raise SystemExit(
                "ERROR: strict mode failed (checkpoint promotion).\n"
                f"  run_dir: {run_dir}\n"
                f"  reason:  {exc}"
            )
        print(f"[WARN] checkpoint promotion skipped: {exc}")
        return payload


def _promote_latest_checkpoint_for_eval(
    run_dir: Path,
    *,
    strict: bool,
) -> dict[str, Any]:
    return _promote_checkpoint(
        run_dir,
        strict=bool(strict),
        step=None,
        purpose="eval",
    )


def _ensure_binary(path: Path, make_target: str) -> None:
    if path.exists():
        return
    _run(["make", "--no-print-directory", make_target], cwd=ROOT)
    if not path.exists():
        raise RuntimeError(f"expected binary after build: {path}")


def _run_sample_packer(
    dataset_path: Path,
    tokenizer_json: Path,
    tokenizer_bin: Path,
    seq_len: int,
    out_token_file: Path,
    out_report_json: Path,
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    if not PACK_TOKENS.exists():
        raise RuntimeError(f"sample packer script missing: {PACK_TOKENS}")
    cmd = [
        _python_exec(),
        str(PACK_TOKENS),
        "--dataset",
        str(dataset_path),
        "--tokenizer-lib",
        str(TOKENIZER_LIB),
        "--tokenizer-bin",
        str(tokenizer_bin),
        "--tokenizer-json",
        str(tokenizer_json),
        "--seq-len",
        str(int(seq_len)),
        "--out",
        str(out_token_file),
        "--report-json",
        str(out_report_json),
    ]
    _run(cmd, cwd=ROOT)
    report = _load_json(out_report_json)
    stats = report.get("stats") if isinstance(report, dict) else None
    if not isinstance(stats, dict):
        raise RuntimeError(f"invalid sample pack report: {out_report_json}")
    return out_token_file, report, stats


def _write_svg_dataset(path: Path, repeats: int) -> None:
    lines = [SVG_LINE for _ in range(max(1, int(repeats)))]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _read_non_empty_rows(path: Path) -> list[str]:
    try:
        raw = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        raw = path.read_text(encoding="utf-8", errors="ignore")
    out: list[str] = []
    for line in raw.splitlines():
        if not line.strip():
            continue
        out.append(line.rstrip("\r\n"))
    return out


def _resolve_pad_token_id(run_dir: Path, default: int = 0) -> int:
    # Prefer explicit ids in config.json when present.
    cfg = run_dir / "config.json"
    if cfg.exists():
        try:
            doc = _load_json(cfg)
        except Exception:
            doc = {}
        for key in ("pad_token_id", "eos_token_id", "bos_token_id"):
            v = doc.get(key) if isinstance(doc, dict) else None
            if isinstance(v, int) and v >= 0:
                return int(v)

    # Fallback to tokenizer.json special token table.
    tok = run_dir / "tokenizer.json"
    if tok.exists():
        try:
            tdoc = _load_json(tok)
        except Exception:
            tdoc = {}
        added = tdoc.get("added_tokens") if isinstance(tdoc, dict) else None
        if isinstance(added, list):
            wanted = (
                "<|pad|>",
                "<pad>",
                "<|eos|>",
                "</s>",
                "<|bos|>",
                "<s>",
            )
            for name in wanted:
                for row in added:
                    if not isinstance(row, dict):
                        continue
                    if str(row.get("content") or "") != name:
                        continue
                    tid = row.get("id")
                    if isinstance(tid, int) and tid >= 0:
                        return int(tid)
        model = tdoc.get("model") if isinstance(tdoc, dict) else None
        vocab = model.get("vocab") if isinstance(model, dict) else None
        if isinstance(vocab, dict):
            for name in ("<|pad|>", "<pad>", "<|eos|>", "</s>", "<|bos|>", "<s>"):
                tid = vocab.get(name)
                if isinstance(tid, int) and tid >= 0:
                    return int(tid)

    return int(default)


def _pack_rows_to_seq_windows(
    row_token_ids: list[list[int]],
    seq_len: int,
    pad_token_id: int,
) -> tuple[list[int], dict[str, Any]]:
    """Pack complete rows into fixed seq windows without cross-row bleed.

    Rows are appended greedily into a window until the next row would overflow.
    The remaining slots are padded, and overflow row starts next window.
    """
    seq_len_i = max(1, int(seq_len))
    pad_id = int(pad_token_id)
    total_rows = len(row_token_ids)
    if total_rows == 0:
        raise RuntimeError("sample-aware packing requires at least one non-empty row")

    packed: list[int] = []
    windows = 0
    pad_tokens = 0
    current: list[int] = []
    max_row = 0
    min_row = None
    used_tokens = 0
    for idx, row in enumerate(row_token_ids, start=1):
        n = len(row)
        if n <= 0:
            continue
        max_row = max(max_row, n)
        min_row = n if min_row is None else min(min_row, n)
        if n > seq_len_i:
            raise RuntimeError(
                f"row {idx} has {n} tokens, exceeds seq_len={seq_len_i}; "
                "increase --seq-len or shorten this sample"
            )
        if len(current) + n <= seq_len_i:
            current.extend(row)
            used_tokens += n
            continue
        pad_n = seq_len_i - len(current)
        if pad_n > 0:
            current.extend([pad_id] * pad_n)
            pad_tokens += pad_n
        packed.extend(current)
        windows += 1
        current = list(row)
        used_tokens += n

    if current:
        pad_n = seq_len_i - len(current)
        if pad_n > 0:
            current.extend([pad_id] * pad_n)
            pad_tokens += pad_n
        packed.extend(current)
        windows += 1

    fill_ratio = float(used_tokens) / float(max(1, windows * seq_len_i))
    stats: dict[str, Any] = {
        "mode": "sample",
        "rows": int(total_rows),
        "windows": int(windows),
        "seq_len": int(seq_len_i),
        "pad_token_id": int(pad_id),
        "pad_tokens": int(pad_tokens),
        "used_tokens": int(used_tokens),
        "packed_tokens": int(len(packed)),
        "fill_ratio": float(fill_ratio),
        "min_row_tokens": int(min_row or 0),
        "max_row_tokens": int(max_row),
    }
    return packed, stats


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


# ── Run Ledger utilities ────────────────────────────────────────────────────
# run_ledger.jsonl: append-only execution log written to run_dir (not work_dir).
# Each line is a JSON object. Last record per run_id is authoritative.

def _read_ledger(run_dir: Path) -> list[dict[str, Any]]:
    """Read run_dir/run_ledger.jsonl → list of last-record-per-run_id sorted by run_order."""
    ledger_path = run_dir / "run_ledger.jsonl"
    if not ledger_path.exists():
        return []
    by_run_id: dict[str, dict[str, Any]] = {}
    try:
        for line in ledger_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if isinstance(rec, dict) and rec.get("run_id"):
                by_run_id[str(rec["run_id"])] = rec
    except OSError:
        return []
    return sorted(by_run_id.values(), key=lambda r: int(r.get("run_order") or 0))


def _append_ledger_entry(run_dir: Path, entry: dict[str, Any]) -> None:
    """Append a single JSON record to run_dir/run_ledger.jsonl (never rewrites)."""
    ledger_path = run_dir / "run_ledger.jsonl"
    line = json.dumps(entry, separators=(",", ":")) + "\n"
    with ledger_path.open("a", encoding="utf-8") as fh:
        fh.write(line)


def _build_ledger_start_record(
    run_dir: Path,
    work_dir: Path,
    args: "argparse.Namespace",
    active_stage: str,
) -> dict[str, Any]:
    """Build a status=running ledger record for the current training invocation."""
    existing = _read_ledger(run_dir)
    run_order = len(existing)

    stage_id = _normalize_stage_name(str(getattr(args, "curriculum_stage", "auto") or "auto"))
    if not stage_id or stage_id == "auto":
        stage_id = active_stage or "pretrain"

    stage_pass = sum(
        1 for r in existing
        if r.get("stage_id") == stage_id and str(r.get("status") or "") in {"running", "completed"}
    ) + 1

    phase_label = f"{stage_id}_{stage_pass}"

    dataset_path_raw = str(getattr(args, "data", None) or "")
    dataset_name = Path(dataset_path_raw).name if dataset_path_raw else None

    return {
        "schema": "ck.run_ledger.v1",
        "run_order": run_order,
        "run_id": work_dir.name,
        "stage_id": stage_id,
        "stage_pass": stage_pass,
        "phase_label": phase_label,
        "status": "running",
        "dataset": dataset_path_raw or None,
        "dataset_name": dataset_name,
        "lr": float(getattr(args, "lr", 0) or 0) or None,
        "seq_len": int(getattr(args, "seq_len", 0) or 0) or None,
        "total_tokens": None,
        "steps": None,
        "pack_mode": str(getattr(args, "pack_mode", None) or "") or None,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "ended_at": None,
        "loss_first": None,
        "loss_final": None,
        "loss_min": None,
        "loss_min_step": None,
        "checkpoint_step": None,
        "checkpoint_bump": None,
        "checkpoint_manifest": None,
        "work_dir": str(work_dir),
    }


def _read_bpe_meta_max_piece_bytes(bin_dir: Path | None) -> int | None:
    if bin_dir is None:
        return None
    meta_path = bin_dir / "tokenizer_meta.json"
    if not meta_path.exists():
        return None
    try:
        meta = _load_json(meta_path)
    except Exception:
        return None
    v = meta.get("max_piece_bytes") if isinstance(meta, dict) else None
    if isinstance(v, int) and v >= 0:
        return int(v)
    return None


def _resolve_coverage_manifest_path(
    dataset_path: Path,
    spec_catalog: str | None,
) -> Path | None:
    candidates: list[Path] = []
    explicit: Path | None = None
    if isinstance(spec_catalog, str) and spec_catalog.strip():
        explicit = Path(spec_catalog).expanduser().resolve()
        if explicit.suffix == ".json" and explicit.name.endswith("_coverage_manifest.json"):
            if explicit.exists():
                return explicit
            return explicit

    parent = dataset_path.parent
    if not parent.exists():
        return explicit
    candidates.extend(sorted(parent.glob("*_coverage_manifest.json")))
    if not candidates:
        return explicit

    # Prefer stage-aware manifests first, then same-prefix manifests.
    stem = dataset_path.stem.lower()
    stage_hint: str | None = None
    if "stage_a" in stem or "pretrain_a" in stem:
        stage_hint = "stage_a"
    elif "stage_b" in stem or "pretrain_b" in stem or "midtrain" in stem:
        stage_hint = "stage_b"
    elif "sft" in stem:
        stage_hint = "sft"

    if stage_hint:
        stage_matches = [p for p in candidates if stage_hint in p.name.lower()]
        if stage_matches:
            candidates = stage_matches

    prefix = stem.split("_instruction_", 1)[0]
    prefix = prefix.split("_svg_", 1)[0]
    if prefix:
        prefix_matches = [p for p in candidates if p.name.lower().startswith(prefix)]
        if prefix_matches:
            candidates = prefix_matches

    candidates.sort(key=lambda p: p.stat().st_mtime if p.exists() else 0.0, reverse=True)
    return candidates[0]


def _check_coverage_gate(
    *,
    dataset_path: Path,
    spec_catalog: str | None,
    strict: bool,
) -> dict[str, Any]:
    # Tokenizer corpora may aggregate multiple stages; enforce all matching
    # stage coverage manifests instead of picking one by mtime.
    dataset_stem = dataset_path.stem.lower()
    if "tokenizer_corpus" in dataset_stem:
        parent = dataset_path.parent
        candidates = sorted(parent.glob("*_coverage_manifest.json")) if parent.exists() else []
        prefix = dataset_stem.split("_tokenizer_corpus", 1)[0]
        if prefix:
            pref = [p for p in candidates if p.name.lower().startswith(prefix)]
            if pref:
                candidates = pref
        if candidates:
            failures: list[str] = []
            checked: list[str] = []
            for manifest in candidates:
                checked.append(str(manifest))
                try:
                    payload = _load_json(manifest)
                except Exception as exc:
                    failures.append(f"{manifest.name}: invalid_manifest:{exc}")
                    continue
                gate = payload.get("gate")
                if not isinstance(gate, dict):
                    failures.append(f"{manifest.name}: missing_gate_block")
                    continue
                if not bool(gate.get("passed")):
                    local_failures = list(gate.get("failures") or [])
                    if local_failures:
                        for item in local_failures:
                            failures.append(f"{manifest.name}: {item}")
                    else:
                        failures.append(f"{manifest.name}: gate_failed")
            if failures and strict:
                raise SystemExit(
                    "ERROR: strict coverage gate failed for tokenizer corpus aggregate.\n"
                    f"  dataset: {dataset_path}\n"
                    "  failures:\n  - "
                    + "\n  - ".join([str(x) for x in failures])
                )
            return {
                "status": "ok" if not failures else "warn",
                "passed": len(failures) == 0,
                "failures": failures,
                "manifest_path": str(candidates[-1]),
                "checked_manifests": checked,
            }

    manifest_path = _resolve_coverage_manifest_path(dataset_path, spec_catalog)
    if manifest_path is None:
        if strict:
            raise SystemExit(
                "ERROR: strict coverage gate enabled but no coverage manifest found.\n"
                f"  dataset: {dataset_path}\n"
                "  expected: *_coverage_manifest.json beside dataset"
            )
        return {"status": "skipped", "reason": "manifest_not_found", "manifest_path": None}

    if not manifest_path.exists():
        if strict:
            raise SystemExit(
                "ERROR: strict coverage gate enabled but coverage manifest path is missing.\n"
                f"  manifest: {manifest_path}"
            )
        return {"status": "skipped", "reason": "manifest_path_missing", "manifest_path": str(manifest_path)}

    payload: dict[str, Any]
    try:
        payload = _load_json(manifest_path)
    except Exception as exc:
        if strict:
            raise SystemExit(
                "ERROR: strict coverage gate failed to read coverage manifest.\n"
                f"  manifest: {manifest_path}\n"
                f"  reason: {exc}"
            )
        return {
            "status": "warn",
            "reason": f"invalid_manifest:{exc}",
            "manifest_path": str(manifest_path),
        }

    gate = payload.get("gate")
    if not isinstance(gate, dict):
        if strict:
            raise SystemExit(
                "ERROR: strict coverage gate found manifest without gate block.\n"
                f"  manifest: {manifest_path}"
            )
        return {"status": "warn", "reason": "missing_gate_block", "manifest_path": str(manifest_path)}

    passed = bool(gate.get("passed"))
    failures = list(gate.get("failures") or [])
    if not passed and strict:
        raise SystemExit(
            "ERROR: strict coverage gate failed.\n"
            f"  manifest: {manifest_path}\n"
            "  failures:\n  - "
            + "\n  - ".join([str(x) for x in failures])
        )
    return {
        "status": "ok" if passed else "warn",
        "passed": passed,
        "failures": failures,
        "manifest_path": str(manifest_path),
    }


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


def _encode_segment_with_bpe_fallback(handle: _TrueBPEHandle, seg: str, chunk_chars: int = 8192) -> list[int]:
    """Encode one segment with split fallback when runtime returns 0 ids."""
    if not seg:
        return []
    ids = handle.encode(seg)
    if len(ids) > 0:
        return ids
    if len(seg) <= 1:
        raise RuntimeError("BPE encoding produced 0 ids for non-empty 1-char segment.")

    # Split and retry to avoid dropping content on long/complex lines.
    step = max(1, min(int(chunk_chars), len(seg) // 2))
    out: list[int] = []
    for i in range(0, len(seg), step):
        chunk = seg[i : i + step]
        if not chunk:
            continue
        chunk_ids = handle.encode(chunk)
        if len(chunk_ids) > 0:
            out.extend(chunk_ids)
            continue
        if len(chunk) <= 1:
            raise RuntimeError("BPE encoding produced 0 ids for fallback chunk.")
        half = max(1, len(chunk) // 2)
        for j in range(0, len(chunk), half):
            piece = chunk[j : j + half]
            if not piece:
                continue
            piece_ids = handle.encode(piece)
            if len(piece_ids) == 0:
                raise RuntimeError(
                    f"BPE encoding produced 0 ids for fallback piece (len={len(piece)})."
                )
            out.extend(piece_ids)
    return out


def _encode_large_text_with_bpe_handle(handle: _TrueBPEHandle, text: str, chunk_chars: int = 8192) -> list[int]:
    """
    Encode long corpora robustly without silently dropping segments.

    Some tokenizer runtimes can return 0 ids for large single-buffer input.
    We preserve text content by splitlines(keepends=True) and retrying with
    progressively smaller chunks when needed.
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
        out.extend(_encode_segment_with_bpe_fallback(handle, seg, chunk_chars=chunk_chars))
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


def _atomic_copy(src: Path, dst: Path) -> None:
    """Copy file atomically via temp file + rename."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(dst.parent), suffix=".tmp")
    try:
        os.close(fd)
        fd = -1
        if os.path.exists(tmp):
            os.unlink(tmp)
        shutil.copy2(src, tmp)
        os.replace(tmp, str(dst))
    except BaseException:
        if fd >= 0:
            os.close(fd)
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def _snapshot_run_checkpoint(run_dir: Path, work_dir: Path, steps: Any) -> dict[str, Any]:
    """Persist a per-run checkpoint snapshot under work_dir for deterministic promotion."""
    try:
        step_i = int(steps)
    except Exception:
        return {"status": "skipped", "reason": "invalid_step", "step": None}
    if step_i <= 0:
        return {"status": "skipped", "reason": "non_positive_step", "step": step_i}

    ckpt_dir = run_dir / "checkpoints"
    src_bump = ckpt_dir / f"weights_step_{step_i:08d}.bump"
    src_manifest = ckpt_dir / f"weights_step_{step_i:08d}_manifest.json"
    if not src_bump.exists() or not src_manifest.exists():
        return {
            "status": "missing",
            "reason": "checkpoint_pair_not_found",
            "step": step_i,
            "source_bump": str(src_bump),
            "source_manifest": str(src_manifest),
        }

    dst_bump = work_dir / "weights_final.bump"
    dst_manifest = work_dir / "weights_final_manifest.json"
    try:
        _atomic_copy(src_bump, dst_bump)
        _atomic_copy(src_manifest, dst_manifest)
    except Exception as exc:
        return {
            "status": "error",
            "reason": f"copy_failed:{exc}",
            "step": step_i,
            "source_bump": str(src_bump),
            "source_manifest": str(src_manifest),
        }
    return {
        "status": "ok",
        "reason": "snapshot_saved",
        "step": step_i,
        "source_bump": str(src_bump),
        "source_manifest": str(src_manifest),
        "bump": str(dst_bump),
        "manifest": str(dst_manifest),
    }


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

    def _is_svg_compatible_row(raw: bytes) -> bool:
        probe = raw.lstrip().lower()
        if probe.startswith(b"<svg"):
            return True
        if probe.startswith(b"<task>") and (b"</task>" in probe) and (b"<svg" in probe):
            return True
        svg_pos = probe.find(b"<svg")
        if svg_pos > 0 and probe.startswith(b"[") and b"]" in probe[:svg_pos]:
            return True
        return False

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

        if require_svg_rows and (not _is_svg_compatible_row(stripped)) and len(svg_issues) < max_issues:
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
                "  # For instruction/tag-prefixed rows (<task>...</task><svg...> or [tags]<svg...>), omit --svg-only:",
                f"  python3 version/v7/scripts/prepare_ascii_dataset_v7.py --input {shlex.quote(str(dataset_path))} --output {shlex.quote(str(dataset_path))} --input-format text --ascii-mode xml_escape",
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
        "min_valid_svg_rate": float(getattr(args, "min_valid_svg_rate", 0.70)),
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
    if response_text:
        # ck_chat may emit runtime log lines into stdout near "Response:".
        # Remove known log-line prefixes so SVG parsing sees model text only.
        cleaned_lines = [
            ln for ln in response_text.splitlines()
            if (not ln.startswith("[CK ")) and (not ln.startswith("[OpenMP]"))
        ]
        response_text = "\n".join(cleaned_lines).strip()

    eval_text = response_text
    prompt_norm = prompt.strip()
    if args.require_svg_rows and prompt_norm.lower().startswith("<svg"):
        # Prompt carries the opening tag in the default SVG eval flow.
        # Evaluate SVG fragments on prompt+response so continuation generations
        # are not misclassified as having zero <svg tags.
        eval_text = f"{prompt_norm}{response_text}"

    open_tags = len(re.findall(r"<svg\b", eval_text, flags=re.IGNORECASE))
    close_tags = len(re.findall(r"</svg>", eval_text, flags=re.IGNORECASE))
    svg_fragments = re.findall(r"<svg\b.*?</svg>", eval_text, flags=re.IGNORECASE | re.DOTALL)

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
            "eval_chars": int(len(eval_text)),
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
    threshold = float(getattr(args, "min_valid_svg_rate", 0.70))
    diagnosis: list[str] = []
    if sample_count == 0:
        diagnosis.append("no_complete_svg_output")
    if valid_rate < threshold:
        diagnosis.append("valid_svg_rate_below_threshold")
    if closure_rate < 0.90:
        diagnosis.append("svg_closure_incomplete")
    if loop_score > 0.50:
        diagnosis.append("repetition_or_looping_detected")
    eval_payload["quality_diagnosis"] = diagnosis or ["ok"]
    eval_payload["quality_recommendations"] = [
        "This gate measures output quality and data/task fit, not CK-vs-PyTorch numerical parity.",
        "Increase SVG corpus size/diversity and rebalance frequent templates.",
        "Add instruction-to-SVG SFT pairs so the model learns prompt-conditioned structure.",
        "Run another pretrain+SFT pass; keep strict gate enabled for production quality sign-off.",
    ]
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


def _manifest_rows(payload: dict[str, Any]) -> int | None:
    for key in ("output_rows", "num_rows", "num_train", "num_samples", "rows", "train_rows"):
        val = payload.get(key)
        if isinstance(val, int):
            return int(val)
        if isinstance(val, float) and math.isfinite(val):
            return int(val)
    return None


def _canon_path(path_like: str) -> str:
    try:
        return str(Path(path_like).expanduser().resolve())
    except Exception:
        return str(path_like)


def _dataset_ref_from_provenance(row: dict[str, Any], *, origin: str) -> dict[str, Any]:
    ds_hash = row.get("hash")
    if isinstance(ds_hash, dict):
        ds_hash = ds_hash.get("value")
    elif not isinstance(ds_hash, str):
        ds_hash = None
    return {
        "dataset_name": row.get("dataset_name"),
        "source_path": row.get("source_path"),
        "stage": row.get("stage"),
        "curriculum_stage": row.get("curriculum_stage"),
        "rows": row.get("rows"),
        "token_count": row.get("token_count"),
        "byte_size": row.get("byte_size"),
        "sha256": ds_hash if isinstance(ds_hash, str) and ds_hash else None,
        "origin": origin,
    }


def _build_tokenizer_corpus_contract(
    *,
    run_dir: Path,
    dataset_path: Path,
    active_stage: str,
    curriculum_stage: str,
    active_rows: int | None,
    token_count: int,
    dataset_size: int,
    dataset_hash: str | None,
    reused_run_tokenizer: bool,
    tokenizer_sha256: str | None,
    tokenizer_path: str | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    active_ref: dict[str, Any] = {
        "dataset_name": dataset_path.name,
        "source_path": str(dataset_path),
        "stage": active_stage,
        "curriculum_stage": curriculum_stage,
        "rows": int(active_rows) if isinstance(active_rows, int) else None,
        "token_count": int(token_count),
        "byte_size": int(dataset_size),
        "sha256": dataset_hash,
        "origin": "current_run",
    }

    corpora: list[dict[str, Any]] = []
    coverage_note = ""
    if reused_run_tokenizer:
        prev_path = run_dir / "training_pipeline_latest.json"
        prev_payload: dict[str, Any] = {}
        if prev_path.exists():
            try:
                prev_payload = _load_json(prev_path)
            except Exception:
                prev_payload = {}

        prev_lineage = prev_payload.get("tokenizer_lineage") if isinstance(prev_payload, dict) else {}
        same_tokenizer = True
        if isinstance(prev_lineage, dict):
            prev_sha = prev_lineage.get("tokenizer_sha256")
            prev_tok_path = prev_lineage.get("tokenizer_path")
            if isinstance(tokenizer_sha256, str) and tokenizer_sha256:
                same_tokenizer = bool(isinstance(prev_sha, str) and prev_sha == tokenizer_sha256)
            elif isinstance(tokenizer_path, str) and tokenizer_path and isinstance(prev_tok_path, str):
                same_tokenizer = _canon_path(prev_tok_path) == _canon_path(tokenizer_path)

            prev_corpora = prev_lineage.get("tokenizer_corpora")
            if same_tokenizer and isinstance(prev_corpora, list) and prev_corpora:
                for row in prev_corpora:
                    if isinstance(row, dict):
                        corpora.append(dict(row))

        if not corpora and isinstance(prev_payload, dict):
            prev_prov = prev_payload.get("data_provenance")
            if isinstance(prev_prov, list):
                for row in prev_prov:
                    if isinstance(row, dict):
                        corpora.append(_dataset_ref_from_provenance(row, origin="previous_run_inferred"))
                        break
        if not corpora:
            coverage_note = "tokenizer was reused but prior tokenizer corpus metadata was unavailable"
    else:
        corpora = [active_ref]

    # Deduplicate corpus refs by path/hash/name.
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for row in corpora:
        if not isinstance(row, dict):
            continue
        path_key = _canon_path(str(row.get("source_path") or ""))
        hash_key = str(row.get("sha256") or "")
        name_key = str(row.get("dataset_name") or "")
        key = (path_key, hash_key, name_key)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    corpora = deduped

    active_in_corpus = False
    active_path = _canon_path(str(active_ref.get("source_path") or ""))
    active_hash = str(active_ref.get("sha256") or "")
    for row in corpora:
        r_path = _canon_path(str(row.get("source_path") or ""))
        r_hash = str(row.get("sha256") or "")
        if active_hash and r_hash and active_hash == r_hash:
            active_in_corpus = True
            break
        if active_path and r_path and active_path == r_path:
            active_in_corpus = True
            break

    if active_in_corpus:
        status = "pass"
        note = "active dataset is covered by tokenizer corpus metadata"
    elif reused_run_tokenizer and corpora:
        status = "warn"
        note = "active dataset is not in tokenizer corpus metadata (tokenizer reused)"
    elif reused_run_tokenizer and not corpora:
        status = "unknown"
        note = coverage_note or "tokenizer corpus metadata unavailable"
    else:
        status = "pass"
        note = "tokenizer built in current run from active dataset"

    coverage = {
        "status": status,
        "active_dataset_in_corpus": bool(active_in_corpus),
        "active_dataset_name": active_ref.get("dataset_name"),
        "active_dataset_path": active_ref.get("source_path"),
        "active_dataset_sha256": active_ref.get("sha256"),
        "note": note,
    }
    return corpora, coverage


def _build_stage_dataset_bindings(
    *,
    stage_timeline: list[dict[str, Any]],
    dataset_catalog: list[dict[str, Any]],
    tokenizer_corpora: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    tok_path_set = {
        _canon_path(str(row.get("source_path") or ""))
        for row in tokenizer_corpora
        if isinstance(row, dict) and isinstance(row.get("source_path"), str) and row.get("source_path")
    }
    tok_hash_set = {
        str(row.get("sha256"))
        for row in tokenizer_corpora
        if isinstance(row, dict) and isinstance(row.get("sha256"), str) and row.get("sha256")
    }

    out: list[dict[str, Any]] = []
    for row in stage_timeline:
        stage = str(row.get("stage") or "")
        if not stage:
            continue
        ds_rows: list[dict[str, Any]] = []
        for entry in dataset_catalog:
            if not isinstance(entry, dict):
                continue
            if str(entry.get("stage") or "") != stage:
                continue
            kind = str(entry.get("kind") or "")
            if kind not in {"active_dataset", "generated_dataset"}:
                continue
            path = entry.get("path")
            sha = entry.get("sha256")
            in_tok = False
            if isinstance(sha, str) and sha and sha in tok_hash_set:
                in_tok = True
            elif isinstance(path, str) and path and _canon_path(path) in tok_path_set:
                in_tok = True
            ds_rows.append(
                {
                    "name": entry.get("name"),
                    "path": path,
                    "rows": entry.get("rows"),
                    "kind": kind,
                    "status": entry.get("status"),
                    "source": entry.get("source"),
                    "sha256": sha if isinstance(sha, str) and sha else None,
                    "in_tokenizer_corpus": bool(in_tok),
                }
            )

        rows_total = 0
        rows_known = False
        for ds in ds_rows:
            rv = ds.get("rows")
            if isinstance(rv, int):
                rows_total += int(rv)
                rows_known = True
        out.append(
            {
                "stage": stage,
                "order": row.get("order"),
                "status": row.get("status"),
                "active": bool(row.get("active") is True),
                "datasets": ds_rows,
                "dataset_count": len(ds_rows),
                "rows_total": rows_total if rows_known else None,
                "tokenizer_coverage": {
                    "in_corpus": sum(1 for ds in ds_rows if ds.get("in_tokenizer_corpus") is True),
                    "not_in_corpus": sum(1 for ds in ds_rows if ds.get("in_tokenizer_corpus") is not True),
                },
            }
        )
    return out


def _infer_dataset_stage(name: str, active_stage: str) -> str:
    probe = str(name or "").lower()
    if any(tok in probe for tok in ("dpo",)):
        return "dpo"
    if any(tok in probe for tok in ("grpo",)):
        return "grpo"
    if any(tok in probe for tok in ("ppo", "rl")):
        return "ppo"
    if any(tok in probe for tok in ("stage_b", "midtrain")):
        return "midtrain"
    if any(tok in probe for tok in ("instruction", "sft")):
        return "sft"
    if any(tok in probe for tok in ("stage_a", "bridge", "assets", "ascii", "svg")):
        return "pretrain"
    return active_stage


def _collect_dataset_catalog(
    dataset_path: Path,
    run_dir: Path,
    active_stage: str,
    dataset_qc: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    seen_paths: set[str] = set()

    def _add_entry(
        *,
        stage: str,
        kind: str,
        name: str,
        path: str,
        rows: int | None = None,
        note: str = "",
        status: str = "ready",
        source: str = "local",
        sha256: str | None = None,
    ) -> None:
        key = str(path)
        if key in seen_paths:
            return
        seen_paths.add(key)
        entries.append(
            {
                "stage": str(stage),
                "kind": str(kind),
                "name": str(name),
                "path": key,
                "rows": int(rows) if isinstance(rows, int) else None,
                "note": str(note or ""),
                "status": str(status),
                "source": str(source),
                "sha256": str(sha256) if isinstance(sha256, str) and sha256 else None,
            }
        )

    active_rows = None
    if isinstance(dataset_qc, dict):
        val = dataset_qc.get("non_empty_lines")
        if isinstance(val, int):
            active_rows = int(val)
    active_hash = _sha256_file(dataset_path) if dataset_path.exists() else None
    _add_entry(
        stage=active_stage,
        kind="active_dataset",
        name=dataset_path.name,
        path=str(dataset_path),
        rows=active_rows,
        note="dataset used for this run stage",
        sha256=active_hash,
    )

    manifest_paths: list[Path] = []
    if dataset_path.parent.exists():
        manifest_paths.extend(sorted(dataset_path.parent.glob("*manifest.json")))
    repo_data_dir = ROOT / "version" / "v7" / "data"
    if repo_data_dir.exists():
        manifest_paths.extend(sorted(repo_data_dir.glob("svg*_manifest.json")))
    autopilot_dir = run_dir / "autopilot"
    if autopilot_dir.exists():
        manifest_paths.extend(sorted(autopilot_dir.glob("iter_*/*manifest*.json")))
        manifest_paths.extend(sorted(autopilot_dir.glob("iter_*/**/*manifest*.json")))

    seen_manifest: set[str] = set()
    for manifest_path in manifest_paths:
        key = str(manifest_path)
        if key in seen_manifest:
            continue
        seen_manifest.add(key)
        payload: dict[str, Any] | None = None
        try:
            payload = _load_json(manifest_path)
        except Exception:
            payload = None
        if not isinstance(payload, dict):
            continue

        rows = _manifest_rows(payload)
        out_path = payload.get("out_path")
        ds_name = payload.get("dataset_name")
        if not isinstance(ds_name, str) or not ds_name.strip():
            if isinstance(out_path, str) and out_path.strip():
                ds_name = Path(out_path).name
            else:
                ds_name = manifest_path.stem
        stage = _infer_dataset_stage(manifest_path.name, active_stage)
        note_parts: list[str] = []
        mapped = payload.get("mapped_common_symbols_total")
        if isinstance(mapped, int):
            note_parts.append(f"mapped_symbols={mapped}")
        tc = payload.get("type_counts")
        if isinstance(tc, dict):
            note_parts.append(f"type_families={len(tc)}")
        if isinstance(payload.get("source_files"), int):
            note_parts.append(f"source_files={payload.get('source_files')}")

        _add_entry(
            stage=stage,
            kind="manifest",
            name=ds_name,
            path=str(manifest_path),
            rows=rows,
            note=", ".join(note_parts),
            source="manifest",
        )
        if isinstance(out_path, str) and out_path.strip():
            _add_entry(
                stage=stage,
                kind="generated_dataset",
                name=Path(out_path).name,
                path=out_path,
                rows=rows,
                note=f"derived from {manifest_path.name}",
                source="manifest_output",
            )

    # Auto-discover prepared .txt datasets in the data directory for planned future stages.
    # This pre-populates midtrain/sft/etc. bindings so the operator can see what datasets
    # are ready without having to formally run each stage first.
    data_dir = dataset_path.parent
    if data_dir.exists():
        for _txt in sorted(data_dir.glob("*.txt")):
            if str(_txt) in seen_paths:
                continue
            _nm = _txt.name
            # Skip raw/intermediate variants and per-split files — only the final merged
            # datasets that would actually be trained on.
            if _nm.endswith("_raw.txt"):
                continue
            if any(_sfx in _nm for _sfx in ("_holdout.", "_train.", "_all.")):
                continue
            _inferred = _infer_dataset_stage(_nm, active_stage)
            _add_entry(
                stage=_inferred,
                kind="active_dataset",
                name=_nm,
                path=str(_txt),
                note="discovered — ready for this stage",
                source="local",
            )

    return entries


def _build_stage_artifacts(
    *,
    stage_timeline: list[dict[str, Any]],
    active_stage: str,
    data_provenance: list[dict[str, Any]],
    data_lab: dict[str, Any],
    bpe_artifacts: dict[str, Any],
    run_dir: Path,
) -> list[dict[str, Any]]:
    prov_by_stage: dict[str, dict[str, Any]] = {}
    for row in data_provenance:
        if isinstance(row, dict):
            stage = row.get("stage")
            if isinstance(stage, str) and stage not in prov_by_stage:
                prov_by_stage[stage] = row

    artifacts_map = data_lab.get("artifacts") if isinstance(data_lab.get("artifacts"), dict) else {}
    active_artifacts: list[dict[str, Any]] = []

    def _push(label: str, path: str | None, required: bool = False) -> None:
        if not isinstance(path, str) or not path.strip():
            return
        active_artifacts.append(
            {
                "label": str(label),
                "path": str(path),
                "required": bool(required),
                "exists": bool(Path(path).expanduser().exists()),
            }
        )

    _push("dataset_path", data_lab.get("dataset_path"), required=True)
    _push("tokenizer_json", data_lab.get("tokenizer_json_path"), required=True)
    _push("dataset_qc_json", artifacts_map.get("dataset_qc_json"), required=True)
    _push("dataset_profile_json", artifacts_map.get("dataset_profile_json"), required=True)
    _push("tokenizer_roundtrip_json", artifacts_map.get("tokenizer_roundtrip_json"), required=True)
    _push("post_train_eval_json", artifacts_map.get("post_train_eval_json"), required=False)
    _push("token_file", bpe_artifacts.get("token_file"), required=False)
    _push("tokenizer_binary_dir", bpe_artifacts.get("run_binary_dir") or bpe_artifacts.get("binary_dir"), required=False)
    for label, rel in (
        ("training_loss_curve", "training_loss_curve_latest.json"),
        ("training_pipeline", "training_pipeline_latest.json"),
        ("ir1_train", "ir1_train_forward.json"),
        ("ir2_train", "ir2_train_backward.json"),
        ("layout_train", "layout_train.json"),
        ("train_exec_plan", "train_exec_plan.json"),
    ):
        p = run_dir / rel
        if p.exists():
            _push(label, str(p), required=False)

    stage_rows: list[dict[str, Any]] = []
    for row in stage_timeline:
        stage = str(row.get("stage") or "")
        if not stage:
            continue
        prov = prov_by_stage.get(stage, {})
        stage_rows.append(
            {
                "stage": stage,
                "status": str(row.get("status") or ("active" if stage == active_stage else "planned")),
                "active": bool(row.get("active") is True or stage == active_stage),
                "dataset_name": prov.get("dataset_name"),
                "token_count": prov.get("token_count"),
                "source_path": prov.get("source_path"),
                "artifacts": list(active_artifacts) if stage == active_stage else [],
            }
        )
    return stage_rows


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
    resume_checkpoint: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build ``training_pipeline_latest.json`` in the schema the visualizer expects.

    Schema: ``ck.training_pipeline.v1``  (see ck_run_v7.py _build_training_pipeline_payload).
    """
    now_iso = datetime.now(timezone.utc).isoformat()
    curriculum_raw = str(getattr(args, "curriculum_stage", "auto") or "auto").strip().lower()
    if curriculum_raw in {"stage_b", "midtrain"}:
        active_stage = "midtrain"
        curriculum_stage = "stage_b"
    elif curriculum_raw in {"stage_a", "pretrain"}:
        active_stage = "pretrain"
        curriculum_stage = "stage_a"
    elif curriculum_raw in {"sft"}:
        active_stage = "sft"
        curriculum_stage = "sft"
    elif curriculum_raw in {"dpo"}:
        active_stage = "dpo"
        curriculum_stage = "dpo"
    elif curriculum_raw in {"grpo"}:
        active_stage = "grpo"
        curriculum_stage = "grpo"
    elif curriculum_raw in {"ppo"}:
        active_stage = "ppo"
        curriculum_stage = "ppo"
    else:
        active_stage = "pretrain"
        curriculum_stage = "auto"
    stage_names = ["pretrain", "midtrain", "sft", "dpo", "grpo", "ppo"]
    active_idx = stage_names.index(active_stage) if active_stage in stage_names else 0
    stage_timeline = [
        {
            "stage": s,
            "order": i,
            "status": "active" if s == active_stage else ("planned" if i > active_idx else "completed"),
            "active": s == active_stage,
        }
        for i, s in enumerate(stage_names)
    ]
    stage_sequence = [
        {
            "stage": row["stage"],
            "seq": i + 1,
            "declared_seq": int(row.get("order", i)) + 1,
            "status": row.get("status"),
            "active": bool(row.get("active") is True),
            "source": "stage_timeline",
        }
        for i, row in enumerate(stage_timeline)
        if isinstance(row, dict) and row.get("stage")
    ]

    # ── data provenance ─────────────────────────────────────────
    dataset_hash = _sha256_file(dataset_path) if dataset_path.exists() else None
    dataset_size = dataset_path.stat().st_size if dataset_path.exists() else 0
    token_count = bpe_artifacts.get("token_count") or 0
    if not token_count and dataset_path.exists():
        # byte tokenizer: one token per byte
        token_count = dataset_size

    active_rows = None
    if isinstance(dataset_qc, dict):
        qc_rows = dataset_qc.get("non_empty_lines")
        if isinstance(qc_rows, int):
            active_rows = int(qc_rows)

    data_provenance = [
        {
            "stage": active_stage,
            "curriculum_stage": curriculum_stage,
            "dataset_name": dataset_path.name,
            "source_path": str(dataset_path),
            "split": "train",
            "rows": int(active_rows) if isinstance(active_rows, int) else None,
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
        bpe_meta_cap: int | None = None
        bpe_bin_dir_any = bpe_artifacts.get("run_binary_dir") or bpe_artifacts.get("binary_dir")
        if isinstance(bpe_bin_dir_any, str) and bpe_bin_dir_any:
            bpe_meta_cap = _read_bpe_meta_max_piece_bytes(Path(bpe_bin_dir_any))
        tokenizer_lineage["bpe_vocab_size"] = int(args.bpe_vocab_size)
        tokenizer_lineage["bpe_min_freq"] = int(args.bpe_min_freq)
        tokenizer_lineage["bpe_max_piece_bytes"] = (
            int(bpe_meta_cap) if isinstance(bpe_meta_cap, int) else int(args.bpe_max_piece_bytes)
        )
        tokenizer_lineage["bpe_mode"] = "ascii_bpe" if tokenizer_kind == "ascii_bpe" else "bytelevel_bpe"
        # Operator-visible continuity signal:
        # True => token IDs are frozen from run_dir/tokenizer.json (--reuse-run-tokenizer).
        # False => tokenizer was rebuilt in this pass.
        tokenizer_lineage["reused_run_tokenizer"] = bool(bpe_artifacts.get("reused_run_tokenizer", False))

    reused_run_tokenizer = bool(bpe_artifacts.get("reused_run_tokenizer", False))
    tokenizer_corpora, tokenizer_coverage = _build_tokenizer_corpus_contract(
        run_dir=run_dir,
        dataset_path=dataset_path,
        active_stage=active_stage,
        curriculum_stage=curriculum_stage,
        active_rows=active_rows,
        token_count=int(token_count),
        dataset_size=int(dataset_size),
        dataset_hash=dataset_hash,
        reused_run_tokenizer=reused_run_tokenizer,
        tokenizer_sha256=tokenizer_lineage.get("tokenizer_sha256"),
        tokenizer_path=tokenizer_lineage.get("tokenizer_path"),
    )
    tokenizer_lineage["tokenizer_corpora"] = tokenizer_corpora
    tokenizer_lineage["active_dataset_in_tokenizer_corpus"] = bool(tokenizer_coverage.get("active_dataset_in_corpus"))
    tokenizer_lineage["coverage_status"] = str(tokenizer_coverage.get("status", "unknown"))
    tokenizer_lineage["coverage_note"] = str(tokenizer_coverage.get("note", ""))

    # Upgrade coverage to 'pass' when the tokenizer roundtrip confirms exact coverage.
    # This handles datasets that are proper subsets of the tokenizer corpus (e.g.
    # stage_a_plus_bridge ⊂ tokenizer_corpus).  Roundtrip exact_match is the ground
    # truth: if all bytes round-trip cleanly, there is no tokenizer gap — the file-
    # level corpus check merely couldn't find the active dataset in the corpus file list.
    _rt_upgraded = False
    if not tokenizer_lineage["active_dataset_in_tokenizer_corpus"] and isinstance(tokenizer_roundtrip, dict):
        _rt_exact = tokenizer_roundtrip.get("exact_match") is True
        _rt_byte = float(tokenizer_roundtrip.get("byte_match_rate") or 0) >= 1.0
        if _rt_exact and _rt_byte:
            _rt_upgraded = True
            tokenizer_lineage["active_dataset_in_tokenizer_corpus"] = True
            tokenizer_lineage["coverage_status"] = "pass"
            tokenizer_lineage["coverage_note"] = (
                "tokenizer roundtrip exact match — all bytes in active dataset are covered "
                "(dataset is a subset of the tokenizer corpus; file not in corpus list)"
            )
            tokenizer_coverage["active_dataset_in_corpus"] = True
            tokenizer_coverage["status"] = "pass"
            tokenizer_coverage["note"] = tokenizer_lineage["coverage_note"]

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

    dataset_catalog = _collect_dataset_catalog(
        dataset_path=dataset_path,
        run_dir=run_dir,
        active_stage=active_stage,
        dataset_qc=dataset_qc,
    )
    stage_artifacts = _build_stage_artifacts(
        stage_timeline=stage_timeline,
        active_stage=active_stage,
        data_provenance=data_provenance,
        data_lab=data_lab,
        bpe_artifacts=bpe_artifacts,
        run_dir=run_dir,
    )
    stage_dataset_bindings = _build_stage_dataset_bindings(
        stage_timeline=stage_timeline,
        dataset_catalog=dataset_catalog,
        tokenizer_corpora=tokenizer_corpora,
    )
    # Propagate roundtrip-based coverage upgrade into stage_dataset_bindings so that
    # pipeline.pipeline.stages[*].datasets[*].in_tokenizer_corpus is consistent.
    if _rt_upgraded:
        _active_path_c = _canon_path(str(dataset_path))
        _active_hash_v = str(dataset_hash or "")
        for _bind in stage_dataset_bindings:
            if not isinstance(_bind, dict) or _bind.get("active") is not True:
                continue
            for _ds in (_bind.get("datasets") or []):
                if not isinstance(_ds, dict):
                    continue
                _ds_path = _canon_path(str(_ds.get("path") or ""))
                _ds_hash = str(_ds.get("sha256") or "")
                if (_active_path_c and _ds_path == _active_path_c) or (
                    _active_hash_v and _ds_hash and _ds_hash == _active_hash_v
                ):
                    _ds["in_tokenizer_corpus"] = True
                    _ds["in_tokenizer_corpus_source"] = "roundtrip"
            # Recompute tokenizer_coverage totals for the patched binding.
            _ds_list = [_d for _d in (_bind.get("datasets") or []) if isinstance(_d, dict)]
            if isinstance(_bind.get("tokenizer_coverage"), dict):
                _bind["tokenizer_coverage"]["in_corpus"] = sum(
                    1 for _d in _ds_list if _d.get("in_tokenizer_corpus") is True
                )
                _bind["tokenizer_coverage"]["not_in_corpus"] = sum(
                    1 for _d in _ds_list if _d.get("in_tokenizer_corpus") is not True
                )
    # ── execution ───────────────────────────────────────────────
    steps = ck_loss.get("steps", 0) if isinstance(ck_loss, dict) else 0
    tokens_per_update = int(args.seq_len) * int(args.grad_accum)
    roundtrip_status = str((tokenizer_roundtrip or {}).get("status") or "unknown")
    run_sequence = [
        {
            "order": 1,
            "op": "dataset_qc",
            "status": "done" if isinstance(dataset_qc, dict) and dataset_qc else "unknown",
            "details": {"dataset_path": str(dataset_path)},
        },
        {
            "order": 2,
            "op": "tokenizer_build_or_reuse",
            "status": "reused" if reused_run_tokenizer else "built",
            "details": {
                "tokenizer_type": tokenizer_kind,
                "tokenizer_path": tokenizer_lineage.get("tokenizer_path"),
                "tokenizer_sha256": tokenizer_lineage.get("tokenizer_sha256"),
            },
        },
        {
            "order": 3,
            "op": "tokenizer_roundtrip",
            "status": roundtrip_status,
            "details": {
                "exact_match": bool((tokenizer_roundtrip or {}).get("exact_match", False)),
                "line_exact_match_rate": (tokenizer_roundtrip or {}).get("line_eval", {}).get("exact_match_rate"),
            },
        },
        {
            "order": 4,
            "op": "train",
            "status": "done" if int(steps) > 0 else "missing",
            "details": {"steps": int(steps), "seq_len": int(args.seq_len), "grad_accum": int(args.grad_accum)},
        },
        {
            "order": 5,
            "op": "post_train_eval",
            "status": "done" if isinstance(post_train_eval, dict) and post_train_eval else "skipped",
            "details": {
                "valid_svg_rate": (post_train_eval or {}).get("valid_svg_rate"),
                "closure_success_rate": (post_train_eval or {}).get("closure_success_rate"),
                "loop_score": (post_train_eval or {}).get("repetition_loop_score"),
            },
        },
    ]

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

    model_contract = {
        "family": str(getattr(args, "template", "qwen3") or "qwen3"),
        "layers": int(train_dims.get("num_layers", getattr(args, "layers", 0) or 0)),
        "embed_dim": int(train_dims.get("embed_dim", getattr(args, "embed_dim", 0) or 0)),
        "hidden_dim": int(train_dims.get("hidden_dim", getattr(args, "hidden_dim", 0) or 0)),
        "num_heads": int(train_dims.get("num_heads", getattr(args, "num_heads", 0) or 0)),
        "num_kv_heads": int(train_dims.get("num_kv_heads", getattr(args, "num_kv_heads", 0) or 0)),
        "context_len": int(train_dims.get("context_length", getattr(args, "context_len", 0) or 0)),
        "vocab_size": int(train_dims.get("vocab_size", vocab_size or 0)),
    }

    binding_by_stage = {}
    for bind in stage_dataset_bindings:
        if not isinstance(bind, dict):
            continue
        st = str(bind.get("stage") or "")
        if st:
            binding_by_stage[st] = bind
    tokenizer_inputs = []
    for row in tokenizer_corpora:
        if not isinstance(row, dict):
            continue
        tokenizer_inputs.append(
            {
                "name": row.get("name"),
                "path": row.get("source_path"),
                "rows": row.get("rows"),
                "tokens": row.get("token_count"),
                "bytes": row.get("byte_size"),
                "sha256": row.get("sha256"),
                "source": "tokenizer_lineage.tokenizer_corpora",
            }
        )

    pipeline_stages: list[dict[str, Any]] = [
        {
            "stage": "data_preparation",
            "stage_id": 0,
            "status": "done",
            "type": "dataset_qc_ascii_prepare",
            "datasets": [
                {
                    "name": dataset_path.name,
                    "path": str(dataset_path),
                    "rows": active_rows,
                    "tokens": int(token_count),
                    "bytes": int(dataset_size),
                    "sha256": dataset_hash,
                    "source": "data_provenance",
                }
            ],
            "ops": ["dataset_qc", "dataset_profile", "tokenizer_roundtrip"],
        },
        {
            "stage": "tokenizer",
            "stage_id": 1,
            "status": "reused" if reused_run_tokenizer else "built",
            "type": str(tokenizer_kind),
            "tokenizer": {
                "type": tokenizer_kind,
                "vocab_size": int(vocab_size),
                "path": tokenizer_lineage.get("tokenizer_path"),
                "sha256": tokenizer_lineage.get("tokenizer_sha256"),
                "reused_run_tokenizer": reused_run_tokenizer,
            },
            "datasets": tokenizer_inputs,
            "coverage": tokenizer_coverage,
            "ops": ["tokenizer_build_or_reuse"],
        },
    ]
    training_stage_id = 2
    for row in stage_timeline:
        if not isinstance(row, dict):
            continue
        st = str(row.get("stage") or "")
        if not st:
            continue
        bind = binding_by_stage.get(st, {})
        datasets = bind.get("datasets") if isinstance(bind, dict) else []
        if not isinstance(datasets, list):
            datasets = []
        pipeline_stages.append(
            {
                "stage": st,
                "stage_id": int(training_stage_id),
                "status": str(row.get("status") or "planned"),
                "type": "training_stage",
                "datasets": datasets,
                "tokenizer_coverage": bind.get("tokenizer_coverage") if isinstance(bind, dict) else {},
                "ops": ["train"] if st == active_stage else [],
            }
        )
        training_stage_id += 1

    stage_loss_history = _collect_stage_loss_history(run_dir)

    return {
        "schema": "ck.training_pipeline.v1",
        "generated_at": now_iso,
        "model": model_contract,
        "pipeline": {
            "schema": "ck.training_pipeline_contract.v1",
            "source_of_truth": "training_pipeline_latest.json",
            "active_stage": active_stage,
            "stages": pipeline_stages,
        },
        "active_stage": active_stage,
        "curriculum_stage": curriculum_stage,
        "stage_timeline": stage_timeline,
        "stage_sequence": {
            "schema": "ck.stage_sequence.v1",
            "active_stage": active_stage,
            "entries": stage_sequence,
        },
        "stage_artifacts": stage_artifacts,
        "backend": "ck",
        "optimizer": {
            "name": str(args.optimizer),
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
            "curriculum_stage": curriculum_stage,
        },
        "train_dims": train_dims,
        "data_provenance": data_provenance,
        "dataset_catalog": dataset_catalog,
        "stage_dataset_bindings": stage_dataset_bindings,
        "run_sequence": run_sequence,
        "tokenizer_dataset_coverage": tokenizer_coverage,
        "tokenizer_lineage": tokenizer_lineage,
        "data_lab": data_lab,
        "stage_loss_history": stage_loss_history,
        "sources": {
            "summary": "train_data_pipeline_v7",
            "run_dir": str(run_dir),
            "resume_checkpoint": dict(resume_checkpoint or {}),
        },
    }


def _backfill_run_ledger(run_dir: Path) -> int:
    """Backfill run_ledger.jsonl for existing .ck_pipeline runs that lack a ledger entry.

    For each run in .ck_pipeline/ that is not already in the ledger, appends a
    completed record derived from pipeline_report.json + train_ck.json. Assigns
    run_order based on file mtime sort of existing unledgered runs.

    Returns the number of records appended.
    """
    pipeline_root = run_dir / ".ck_pipeline"
    if not pipeline_root.exists():
        return 0

    existing_ledger = _read_ledger(run_dir)
    ledgered_ids: set[str] = {r["run_id"] for r in existing_ledger if isinstance(r.get("run_id"), str)}

    # Collect unledgered runs, sorted by train_ck.json mtime (best available ordering)
    unledgered: list[tuple[int, Path]] = []
    for sub in pipeline_root.iterdir():
        if not sub.is_dir():
            continue
        if sub.name in ledgered_ids:
            continue
        train_ck_path = sub / "train_ck.json"
        if not train_ck_path.exists():
            continue
        mtime = int(train_ck_path.stat().st_mtime)
        unledgered.append((mtime, sub))
    unledgered.sort(key=lambda t: t[0])

    if not unledgered:
        return 0

    # Count existing completed entries per stage to assign stage_pass
    stage_pass_counter: dict[str, int] = {}
    for r in existing_ledger:
        sid = str(r.get("stage_id") or "")
        if sid and str(r.get("status") or "") in {"running", "completed"}:
            stage_pass_counter[sid] = stage_pass_counter.get(sid, 0) + 1

    base_run_order = len(existing_ledger)
    appended = 0
    for i, (mtime, sub) in enumerate(unledgered):
        train_ck_path = sub / "train_ck.json"
        try:
            train_ck = _load_json(train_ck_path)
        except Exception:
            continue
        if not isinstance(train_ck, dict):
            continue

        # Derive stage using the same full resolution chain as _collect_stage_loss_history.
        stage_id: str | None = None
        dataset_name: str | None = None
        dataset_path_value: str | None = None

        # 1. pipeline_report.json
        report_path = sub / "pipeline_report.json"
        if report_path.exists():
            try:
                report = _load_json(report_path)
                if isinstance(report, dict):
                    cs = str(report.get("curriculum_stage") or "").strip().lower()
                    stage_id = _normalize_stage_name(cs)
                    ds = report.get("dataset")
                    if isinstance(ds, str) and ds:
                        dataset_path_value = ds
                        dataset_name = Path(ds).name
            except Exception:
                pass

        # 2. train_token_pack.json -> actual source dataset name
        if not dataset_name:
            pack_path = sub / "train_token_pack.json"
            if pack_path.exists():
                try:
                    pack = _load_json(pack_path)
                    if isinstance(pack, dict):
                        ds_raw = str(pack.get("dataset") or "").strip()
                        if ds_raw:
                            if dataset_path_value is None:
                                dataset_path_value = ds_raw
                            dataset_name = Path(ds_raw).name
                except Exception:
                    pass

        # 3. Cross-reference dataset_name against training_plan.json
        if not stage_id and dataset_name:
            plan_path = run_dir / "training_plan.json"
            if plan_path.exists():
                try:
                    plan = _load_json(plan_path)
                    if isinstance(plan, dict):
                        for ps in (plan.get("stages") or []):
                            for pds in (ps.get("datasets") or []):
                                pds_name = Path(str(pds.get("name") or pds.get("path") or "")).name
                                if pds_name and pds_name == dataset_name:
                                    stage_id = _normalize_stage_name(str(ps.get("stage") or ""))
                                    break
                            if stage_id:
                                break
                        if not stage_id:
                            active = _normalize_stage_name(str(plan.get("active_stage") or ""))
                            if active:
                                stage_id = active
                except Exception:
                    pass

        # 4. curriculum_stage / source_stage in train_ck (last resort)
        if not stage_id:
            for key in ("curriculum_stage", "source_stage"):
                cs = str(train_ck.get(key) or "").strip().lower()
                stage_id = _normalize_stage_name(cs)
                if stage_id:
                    break

        if not stage_id:
            stage_id = "pretrain"

        stage_pass_counter[stage_id] = stage_pass_counter.get(stage_id, 0) + 1
        stage_pass = stage_pass_counter[stage_id]
        phase_label = f"{stage_id}_{stage_pass}"

        loss_curve = train_ck.get("loss_curve")
        lc_finite = [p for p in (loss_curve or []) if isinstance(p, dict) and isinstance(p.get("loss_ck"), (int, float))]
        loss_vals = [float(p["loss_ck"]) for p in lc_finite]
        loss_first = loss_vals[0] if loss_vals else None
        loss_final = loss_vals[-1] if loss_vals else None
        loss_min = min(loss_vals) if loss_vals else None
        loss_min_step = (loss_vals.index(loss_min) + 1) if loss_min is not None else None
        ended_at = datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat()

        ck_step_i: int | None = None
        try:
            ck_step_i = int(train_ck.get("steps"))
        except Exception:
            ck_step_i = None
        local_bump = sub / "weights_final.bump"
        local_manifest = sub / "weights_final_manifest.json"
        global_bump = (run_dir / "checkpoints" / f"weights_step_{ck_step_i:08d}.bump") if ck_step_i else None
        global_manifest = (
            run_dir / "checkpoints" / f"weights_step_{ck_step_i:08d}_manifest.json"
        ) if ck_step_i else None
        checkpoint_bump: str | None = None
        checkpoint_manifest: str | None = None
        if local_bump.exists() and local_manifest.exists():
            checkpoint_bump = str(local_bump)
            checkpoint_manifest = str(local_manifest)
        elif global_bump is not None and global_manifest is not None and global_bump.exists() and global_manifest.exists():
            checkpoint_bump = str(global_bump)
            checkpoint_manifest = str(global_manifest)

        rec: dict[str, Any] = {
            "schema": "ck.run_ledger.v1",
            "run_order": base_run_order + i,
            "run_id": sub.name,
            "stage_id": stage_id,
            "stage_pass": stage_pass,
            "phase_label": phase_label,
            "status": "completed",
            "dataset": dataset_path_value,
            "dataset_name": dataset_name,
            "lr": train_ck.get("lr"),
            "seq_len": train_ck.get("seq_len"),
            "total_tokens": train_ck.get("total_tokens"),
            "steps": train_ck.get("steps"),
            "pack_mode": None,
            "started_at": None,
            "ended_at": ended_at,
            "loss_first": loss_first,
            "loss_final": loss_final,
            "loss_min": loss_min,
            "loss_min_step": loss_min_step,
            "checkpoint_step": ck_step_i,
            "checkpoint_bump": checkpoint_bump,
            "checkpoint_manifest": checkpoint_manifest,
            "work_dir": str(sub),
            "source": "backfill",
        }
        _append_ledger_entry(run_dir, rec)
        appended += 1

    return appended


def _collect_stage_loss_history(run_dir: Path) -> dict[str, Any]:
    """Scan .ck_pipeline/*/train_ck.json and pipeline_report.json to build
    stage_loss_history for the IR Visualizer stage cards.

    Stage resolution order (most authoritative first):
      1. run_ledger.jsonl entry for this run_id  (primary — written before training starts)
      2. pipeline_report.json curriculum_stage   (legacy/non-ledger runs)
      3. train_token_pack.json → plan cross-ref  (heuristic)
      4. active_stage from training_plan.json    (heuristic)
      5. source_stage/curriculum_stage from train_ck (last resort, can be mislabeled)
      6. default: "pretrain"
    """
    pipeline_root = run_dir / ".ck_pipeline"
    if not pipeline_root.exists():
        return {"entries": []}

    # Build ledger index: run_id → last record (most recent wins per run_id)
    ledger_by_run_id: dict[str, dict[str, Any]] = {
        r["run_id"]: r for r in _read_ledger(run_dir) if isinstance(r.get("run_id"), str)
    }

    entries: list[dict[str, Any]] = []
    for run_dir_entry in sorted(pipeline_root.iterdir()):
        if not run_dir_entry.is_dir():
            continue
        train_ck_path = run_dir_entry / "train_ck.json"
        if not train_ck_path.exists():
            continue
        try:
            train_ck = _load_json(train_ck_path)
        except Exception:
            continue
        if not isinstance(train_ck, dict):
            continue

        run_id_str = str(run_dir_entry.name)
        stage: str | None = None
        dataset_name: str | None = None
        stage_pass: int | None = None
        phase_label: str | None = None
        run_order: int | None = None

        # ── 1. run_ledger.jsonl (primary) ──────────────────────────────────
        led = ledger_by_run_id.get(run_id_str)
        if isinstance(led, dict):
            stage = _normalize_stage_name(str(led.get("stage_id") or ""))
            dataset_name = str(led.get("dataset_name") or "") or None
            stage_pass = led.get("stage_pass")
            phase_label = led.get("phase_label")
            run_order = led.get("run_order")

        # ── 2. pipeline_report.json (legacy authoritative) ─────────────────
        report_path = run_dir_entry / "pipeline_report.json"
        if report_path.exists():
            try:
                report = _load_json(report_path)
                if isinstance(report, dict):
                    if not stage:
                        cs = str(report.get("curriculum_stage") or "").strip().lower()
                        stage = _normalize_stage_name(cs)
                    if not dataset_name:
                        ds = report.get("dataset")
                        if isinstance(ds, str) and ds:
                            dataset_name = Path(ds).name
            except Exception:
                pass

        # ── 3. train_token_pack.json (dataset name fallback) ───────────────
        if not dataset_name:
            pack_path = run_dir_entry / "train_token_pack.json"
            if pack_path.exists():
                try:
                    pack = _load_json(pack_path)
                    if isinstance(pack, dict):
                        ds_raw = str(pack.get("dataset") or "").strip()
                        if ds_raw:
                            dataset_name = Path(ds_raw).name
                except Exception:
                    pass

        # ── 4. Cross-reference dataset name against training_plan.json ─────
        if not stage and dataset_name:
            plan_path = run_dir / "training_plan.json"
            if plan_path.exists():
                try:
                    plan = _load_json(plan_path)
                    if isinstance(plan, dict):
                        for ps in (plan.get("stages") or []):
                            for pds in (ps.get("datasets") or []):
                                pds_name = Path(
                                    str(pds.get("name") or pds.get("path") or "")
                                ).name
                                if pds_name and pds_name == dataset_name:
                                    stage = _normalize_stage_name(str(ps.get("stage") or ""))
                                    break
                            if stage:
                                break
                        if not stage:
                            active = _normalize_stage_name(str(plan.get("active_stage") or ""))
                            if active:
                                stage = active
                except Exception:
                    pass

        # ── 5. source_stage / curriculum_stage inside train_ck (mislabeled) ─
        if not stage:
            for key in ("curriculum_stage", "source_stage"):
                cs = str(train_ck.get(key) or "").strip().lower()
                stage = _normalize_stage_name(cs)
                if stage:
                    break

        # ── 6. Final fallback ──────────────────────────────────────────────
        if not stage:
            stage = "pretrain"

        # Build downsampled loss points from loss_curve
        loss_curve = train_ck.get("loss_curve")
        if not isinstance(loss_curve, list) or len(loss_curve) == 0:
            continue
        finite_points: list[dict[str, Any]] = []
        for p in loss_curve:
            if not isinstance(p, dict):
                continue
            step = p.get("step")
            loss_val = p.get("loss_ck") if p.get("loss_ck") is not None else p.get("loss")
            if isinstance(step, (int, float)) and isinstance(loss_val, (int, float)):
                if float(loss_val) == float(loss_val):  # NaN guard
                    finite_points.append({"step": int(step), "loss": float(loss_val)})
        if not finite_points:
            continue

        # Downsample to ≤200 points so the JSON stays small
        _max_pts = 200
        if len(finite_points) > _max_pts:
            _stride = max(1, len(finite_points) // _max_pts)
            _sampled = finite_points[::_stride]
            if _sampled[-1] is not finite_points[-1]:
                _sampled.append(finite_points[-1])
        else:
            _sampled = finite_points

        first_loss = float(finite_points[0]["loss"])
        final_loss = float(finite_points[-1]["loss"])
        drop_pct = round((first_loss - final_loss) / first_loss * 100.0, 2) if first_loss > 0 else 0.0

        # Sort key: prefer ledger run_order over file mtime for correct multi-pass ordering
        if run_order is not None:
            sort_key = int(run_order)
            mtime = int(train_ck_path.stat().st_mtime)
        else:
            mtime = int(train_ck_path.stat().st_mtime)
            sort_key = mtime
        ended_at = datetime.fromtimestamp(int(train_ck_path.stat().st_mtime), tz=timezone.utc).isoformat()

        entry: dict[str, Any] = {
            "stage": stage,
            "run_id": run_id_str,
            "ended_at": ended_at,
            "steps": int(train_ck.get("steps") or len(finite_points)),
            "lr": train_ck.get("lr"),
            "seq_len": train_ck.get("seq_len"),
            "dataset_name": dataset_name,
            "first_loss": first_loss,
            "final_loss": final_loss,
            "drop_pct": drop_pct,
            "points": _sampled,
            "_sort_key": sort_key,
        }
        if stage_pass is not None:
            entry["stage_pass"] = stage_pass
        if phase_label is not None:
            entry["phase_label"] = phase_label
        if run_order is not None:
            entry["run_order"] = run_order
        entries.append(entry)

    # Sort by ledger run_order when available, then file mtime for legacy runs
    entries.sort(key=lambda e: e.pop("_sort_key", 0))
    return {"entries": entries}


def _normalize_stage_name(raw: Any) -> str | None:
    s = str(raw or "").strip().lower()
    if not s:
        return None
    if s in {"stage_a", "pretrain"}:
        return "pretrain"
    if s in {"stage_b", "midtrain"}:
        return "midtrain"
    if s in {"sft", "dpo", "grpo", "ppo"}:
        return s
    return s


def _default_stage_order() -> list[str]:
    return ["pretrain", "midtrain", "sft", "dpo", "grpo", "ppo"]


def _resolve_stage_order(
    existing_plan: dict[str, Any] | None,
    training_pipeline: dict[str, Any],
) -> list[str]:
    order: list[str] = []
    seen: set[str] = set()

    def _push(stage: Any) -> None:
        st = _normalize_stage_name(stage)
        if not st or st in seen:
            return
        order.append(st)
        seen.add(st)

    if isinstance(existing_plan, dict):
        existing_order = existing_plan.get("stage_order")
        if isinstance(existing_order, list):
            for stage in existing_order:
                _push(stage)
    stage_seq = training_pipeline.get("stage_sequence")
    if isinstance(stage_seq, dict):
        entries = stage_seq.get("entries")
        if isinstance(entries, list):
            ordered = sorted(
                [e for e in entries if isinstance(e, dict)],
                key=lambda e: int(e.get("seq")) if isinstance(e.get("seq"), int) and int(e.get("seq")) > 0 else 10_000,
            )
            for entry in ordered:
                _push(entry.get("stage"))
    for stage in _default_stage_order():
        _push(stage)
    return order


def _discover_pipeline_report_stage_rows(run_dir: Path) -> list[dict[str, Any]]:
    pipeline_root = run_dir / ".ck_pipeline"
    if not pipeline_root.exists():
        return []
    rows: list[dict[str, Any]] = []
    for report_path in sorted(pipeline_root.glob("*/pipeline_report.json")):
        try:
            report = _load_json(report_path)
        except Exception:
            continue
        if not isinstance(report, dict):
            continue
        stage = _normalize_stage_name(report.get("curriculum_stage"))
        if not stage:
            continue
        dataset_path = report.get("dataset")
        dataset_name = Path(str(dataset_path)).name if isinstance(dataset_path, str) and dataset_path else None
        dataset_qc = report.get("dataset_qc") if isinstance(report.get("dataset_qc"), dict) else {}
        dataset_profile = report.get("dataset_profile") if isinstance(report.get("dataset_profile"), dict) else {}
        token_count = dataset_profile.get("token_count")
        byte_size = dataset_qc.get("bytes")
        row_count = dataset_qc.get("non_empty_lines")
        rows.append(
            {
                "stage": stage,
                "dataset": {
                    "name": dataset_name,
                    "path": dataset_path if isinstance(dataset_path, str) else None,
                    "rows": row_count if isinstance(row_count, int) else None,
                    "tokens": token_count if isinstance(token_count, int) else None,
                    "bytes": byte_size if isinstance(byte_size, int) else None,
                    "source": "pipeline_report",
                },
                "report_path": str(report_path),
                "mtime_ns": int(report_path.stat().st_mtime_ns),
            }
        )
    return rows


def _build_or_update_training_plan_payload(
    *,
    run_dir: Path,
    training_pipeline: dict[str, Any],
    existing_plan: dict[str, Any] | None,
) -> dict[str, Any]:
    now_iso = datetime.now(timezone.utc).isoformat()
    stage_order = _resolve_stage_order(existing_plan, training_pipeline)
    order_idx = {s: i for i, s in enumerate(stage_order)}
    active_stage = _normalize_stage_name(training_pipeline.get("active_stage")) or "pretrain"
    discovered_report_rows = _discover_pipeline_report_stage_rows(run_dir)
    if discovered_report_rows:
        latest = max(discovered_report_rows, key=lambda r: int(r.get("mtime_ns", 0)))
        latest_stage = _normalize_stage_name(latest.get("stage"))
        if latest_stage:
            active_stage = latest_stage

    stage_map: dict[str, dict[str, Any]] = {}

    def _ensure_stage(stage: str) -> dict[str, Any]:
        st = _normalize_stage_name(stage) or stage
        row = stage_map.get(st)
        if row is None:
            row = {
                "stage": st,
                "seq": int(order_idx.get(st, len(order_idx)) + 1),
                "status": "planned",
                "datasets": [],
                "runs": [],
            }
            stage_map[st] = row
        return row

    def _dataset_key(row: dict[str, Any]) -> tuple[str, str]:
        p = str(row.get("path") or "").strip()
        n = str(row.get("name") or "").strip().lower()
        return (_canon_path(p) if p else "", n)

    def _upsert_dataset(stage: str, dataset_row: dict[str, Any]) -> None:
        st = _ensure_stage(stage)
        existing_rows = st.get("datasets")
        if not isinstance(existing_rows, list):
            existing_rows = []
            st["datasets"] = existing_rows
        key = _dataset_key(dataset_row)
        if key == ("", ""):
            return
        for cur in existing_rows:
            if not isinstance(cur, dict):
                continue
            if _dataset_key(cur) == key:
                for k, v in dataset_row.items():
                    if v is None:
                        continue
                    if k not in cur or cur.get(k) in (None, "", [], {}):
                        cur[k] = v
                return
        existing_rows.append(dict(dataset_row))

    # Merge existing plan first (preserve operator edits and prior stages).
    if isinstance(existing_plan, dict):
        existing_stages = existing_plan.get("stages")
        if isinstance(existing_stages, list):
            for row in existing_stages:
                if not isinstance(row, dict):
                    continue
                stage = _normalize_stage_name(row.get("stage"))
                if not stage:
                    continue
                st = _ensure_stage(stage)
                seq = row.get("seq")
                if isinstance(seq, int) and seq > 0:
                    st["seq"] = seq
                status = row.get("status")
                if isinstance(status, str) and status.strip():
                    st["status"] = status.strip().lower()
                datasets = row.get("datasets")
                if isinstance(datasets, list):
                    for ds in datasets:
                        if isinstance(ds, dict):
                            _upsert_dataset(stage, ds)
                # Preserve existing runs audit trail so operator edits survive re-runs.
                existing_runs = row.get("runs")
                if isinstance(existing_runs, list) and existing_runs:
                    st["runs"] = list(existing_runs)

    # Ensure all canonical stages exist.
    for s in stage_order:
        _ensure_stage(s)

    # Merge datasets discovered by current pipeline.
    stage_bindings = training_pipeline.get("stage_dataset_bindings")
    if isinstance(stage_bindings, list):
        for bind in stage_bindings:
            if not isinstance(bind, dict):
                continue
            stage = _normalize_stage_name(bind.get("stage"))
            if not stage:
                continue
            datasets = bind.get("datasets")
            if not isinstance(datasets, list):
                continue
            for ds in datasets:
                if not isinstance(ds, dict):
                    continue
                _upsert_dataset(
                    stage,
                    {
                        "name": ds.get("name"),
                        "path": ds.get("path"),
                        "rows": ds.get("rows"),
                        "kind": ds.get("kind"),
                        "source": ds.get("source"),
                        "status": ds.get("status"),
                        "in_tokenizer_corpus": ds.get("in_tokenizer_corpus"),
                    },
                )

    # Merge active provenance rows as source-of-truth for "what just trained".
    data_provenance = training_pipeline.get("data_provenance")
    if isinstance(data_provenance, list):
        for row in data_provenance:
            if not isinstance(row, dict):
                continue
            stage = _normalize_stage_name(row.get("stage") or row.get("curriculum_stage"))
            if not stage:
                continue
            _upsert_dataset(
                stage,
                {
                    "name": row.get("dataset_name"),
                    "path": row.get("source_path"),
                    "rows": row.get("rows"),
                    "tokens": row.get("token_count"),
                    "bytes": row.get("byte_size"),
                    "source": "data_provenance",
                },
            )

    # Merge stage artifacts (for older runs with sparse provenance).
    stage_artifacts = training_pipeline.get("stage_artifacts")
    if isinstance(stage_artifacts, list):
        for row in stage_artifacts:
            if not isinstance(row, dict):
                continue
            stage = _normalize_stage_name(row.get("stage"))
            if not stage:
                continue
            _upsert_dataset(
                stage,
                {
                    "name": row.get("dataset_name"),
                    "path": row.get("source_path"),
                    "tokens": row.get("token_count"),
                    "source": "stage_artifacts",
                },
            )

    # Merge historical pipeline reports to backfill stage coverage.
    for row in discovered_report_rows:
        stage = _normalize_stage_name(row.get("stage"))
        dataset = row.get("dataset")
        if not stage or not isinstance(dataset, dict):
            continue
        _upsert_dataset(stage, dataset)

    # Compute status from evidence (active > completed > planned).
    timeline = training_pipeline.get("stage_timeline")
    timeline_status: dict[str, str] = {}
    if isinstance(timeline, list):
        for row in timeline:
            if not isinstance(row, dict):
                continue
            stage = _normalize_stage_name(row.get("stage"))
            if not stage:
                continue
            status = str(row.get("status") or "").strip().lower()
            if status:
                timeline_status[stage] = status

    loss_history = training_pipeline.get("stage_loss_history")
    loss_runs: dict[str, int] = {}
    if isinstance(loss_history, dict):
        entries = loss_history.get("entries")
        if isinstance(entries, list):
            for row in entries:
                if not isinstance(row, dict):
                    continue
                stage = _normalize_stage_name(row.get("stage"))
                if stage:
                    loss_runs[stage] = int(loss_runs.get(stage, 0)) + 1

    active_idx = int(order_idx.get(active_stage, 0))
    for stage, st in stage_map.items():
        datasets = st.get("datasets")
        ds_count = len(datasets) if isinstance(datasets, list) else 0
        hint = str(timeline_status.get(stage, st.get("status") or "")).strip().lower()
        idx = int(order_idx.get(stage, len(order_idx)))
        completed_hint = (
            hint in {"completed", "pass"}
            or int(loss_runs.get(stage, 0)) > 0
            or (idx < active_idx and ds_count > 0)
        )
        if stage == active_stage:
            st["status"] = "active"
        elif completed_hint:
            st["status"] = "completed"
        else:
            st["status"] = "planned"
        st["dataset_count"] = ds_count

    # Populate per-stage runs audit trail from .ck_pipeline scan.
    # _collect_stage_loss_history already reads pipeline_report.json for authoritative
    # curriculum_stage labels so the run→stage mapping is always correct.
    raw_history = _collect_stage_loss_history(run_dir)
    _runs_by_stage: dict[str, list[dict[str, Any]]] = {}
    for _entry in raw_history.get("entries", []):
        _s = _normalize_stage_name(_entry.get("stage"))
        if _s:
            _runs_by_stage.setdefault(_s, []).append(_entry)
    for _stage, _st in stage_map.items():
        _history = _runs_by_stage.get(_stage, [])
        if not _history:
            continue
        # Merge by run_id: preserve any operator-added fields from existing plan runs.
        _runs_map: dict[str, dict[str, Any]] = {}
        for _r in (_st.get("runs") or []):
            if isinstance(_r, dict) and _r.get("run_id"):
                _runs_map[str(_r["run_id"])] = dict(_r)
        for _e in _history:
            _rid = _e.get("run_id")
            if not _rid:
                continue
            _rec: dict[str, Any] = {
                "run_id": str(_rid),
                "dataset": _e.get("dataset_name"),
                "lr": _e.get("lr"),
                "steps": _e.get("steps"),
                "first_loss": _e.get("first_loss"),
                "final_loss": _e.get("final_loss"),
                "drop_pct": _e.get("drop_pct"),
                "ended_at": _e.get("ended_at"),
            }
            _cur = _runs_map.get(str(_rid), {})
            for _k, _v in _rec.items():
                if _v is not None:
                    _cur[_k] = _v
            _runs_map[str(_rid)] = _cur
        _merged = sorted(_runs_map.values(), key=lambda r: str(r.get("ended_at") or ""))
        _st["runs"] = _merged
        _st["run_count"] = len(_merged)

    stages_sorted = sorted(
        stage_map.values(),
        key=lambda r: (
            int(r.get("seq")) if isinstance(r.get("seq"), int) else int(order_idx.get(str(r.get("stage")), 999) + 1),
            int(order_idx.get(str(r.get("stage")), 999)),
        ),
    )

    lineage = training_pipeline.get("tokenizer_lineage")
    tokenizer = lineage if isinstance(lineage, dict) else {}
    plan = {
        "schema": "ck.training_plan.v1",
        "created_at": (
            str(existing_plan.get("created_at"))
            if isinstance(existing_plan, dict) and isinstance(existing_plan.get("created_at"), str)
            else now_iso
        ),
        "updated_at": now_iso,
        "run_dir": str(run_dir),
        "active_stage": active_stage,
        "stage_order": stage_order,
        "tokenizer": {
            "type": tokenizer.get("type"),
            "vocab_size": tokenizer.get("vocab_size"),
            "tokenizer_path": tokenizer.get("tokenizer_path"),
            "tokenizer_sha256": tokenizer.get("tokenizer_sha256"),
            "reused_run_tokenizer": tokenizer.get("reused_run_tokenizer"),
            "tokenizer_corpora": tokenizer.get("tokenizer_corpora") if isinstance(tokenizer.get("tokenizer_corpora"), list) else [],
        },
        "stages": stages_sorted,
        "source_pipeline": "training_pipeline_latest.json",
    }
    # Preserve operator-written roadmap across pipeline re-runs.
    if isinstance(existing_plan, dict) and isinstance(existing_plan.get("roadmap"), dict):
        plan["roadmap"] = existing_plan["roadmap"]
    return plan


def _apply_training_plan_to_pipeline_payload(
    training_pipeline: dict[str, Any],
    training_plan: dict[str, Any],
) -> None:
    if not isinstance(training_plan, dict):
        return
    stages = training_plan.get("stages")
    if not isinstance(stages, list):
        return
    active_stage = _normalize_stage_name(training_plan.get("active_stage")) or _normalize_stage_name(training_pipeline.get("active_stage")) or "pretrain"

    stage_rows: list[dict[str, Any]] = []
    stage_seq: list[dict[str, Any]] = []
    stage_bindings: list[dict[str, Any]] = []
    for idx, row in enumerate(stages):
        if not isinstance(row, dict):
            continue
        stage = _normalize_stage_name(row.get("stage"))
        if not stage:
            continue
        seq = row.get("seq")
        if not isinstance(seq, int) or seq <= 0:
            seq = idx + 1
        status = str(row.get("status") or ("active" if stage == active_stage else "planned")).strip().lower()
        active = bool(stage == active_stage or row.get("active") is True)
        datasets = row.get("datasets")
        if not isinstance(datasets, list):
            datasets = []
        rows_total = 0
        rows_known = False
        in_tok = 0
        not_in_tok = 0
        norm_ds: list[dict[str, Any]] = []
        for ds in datasets:
            if not isinstance(ds, dict):
                continue
            rv = ds.get("rows")
            if isinstance(rv, int):
                rows_total += int(rv)
                rows_known = True
            in_corpus = bool(ds.get("in_tokenizer_corpus") is True)
            if in_corpus:
                in_tok += 1
            else:
                not_in_tok += 1
            norm_ds.append(
                {
                    "name": ds.get("name"),
                    "path": ds.get("path"),
                    "rows": rv if isinstance(rv, int) else None,
                    "kind": ds.get("kind"),
                    "status": ds.get("status"),
                    "source": ds.get("source"),
                    "sha256": ds.get("sha256") if isinstance(ds.get("sha256"), str) and ds.get("sha256") else None,
                    "in_tokenizer_corpus": in_corpus,
                    "tokens": ds.get("tokens"),
                    "bytes": ds.get("bytes"),
                }
            )
        stage_rows.append(
            {
                "stage": stage,
                "order": int(seq) - 1,
                "status": status,
                "active": active,
            }
        )
        stage_seq.append(
            {
                "stage": stage,
                "seq": int(seq),
                "declared_seq": int(seq),
                "status": status,
                "active": active,
                "source": "training_plan",
            }
        )
        stage_bindings.append(
            {
                "stage": stage,
                "order": int(seq) - 1,
                "status": status,
                "active": active,
                "datasets": norm_ds,
                "dataset_count": len(norm_ds),
                "rows_total": rows_total if rows_known else None,
                "tokenizer_coverage": {
                    "in_corpus": in_tok,
                    "not_in_corpus": not_in_tok,
                },
            }
        )

    if stage_rows:
        training_pipeline["active_stage"] = active_stage
        training_pipeline["stage_timeline"] = stage_rows
        training_pipeline["stage_sequence"] = {
            "schema": "ck.stage_sequence.v1",
            "active_stage": active_stage,
            "entries": stage_seq,
        }
        training_pipeline["stage_dataset_bindings"] = stage_bindings


def _apply_roundtrip_coverage_to_pipeline(
    training_pipeline: dict[str, Any],
    tokenizer_roundtrip: dict[str, Any] | None,
    dataset_path: Path,
) -> None:
    """Propagate roundtrip-based tokenizer coverage into stage_dataset_bindings and
    pipeline.stages AFTER _apply_training_plan_to_pipeline_payload has run.

    When the roundtrip confirms exact coverage but the file-path/hash check could not
    match the active dataset to the tokenizer corpus (e.g. dataset is a subset of the
    corpus), this function patches all in_tokenizer_corpus flags and coverage counters
    so the visualizer does not show a spurious tok-gap warning.
    """
    if not isinstance(training_pipeline, dict):
        return
    lin = training_pipeline.get("tokenizer_lineage")
    if not isinstance(lin, dict):
        return
    # Only apply when tokenizer_lineage says coverage is pass AND the roundtrip confirms it.
    if lin.get("coverage_status") != "pass":
        return
    if lin.get("active_dataset_in_tokenizer_corpus") is not True:
        return
    rt = tokenizer_roundtrip or {}
    if not (rt.get("exact_match") is True and float(rt.get("byte_match_rate") or 0) >= 1.0):
        return
    active_path_c = _canon_path(str(dataset_path))

    # Patch stage_dataset_bindings.
    for bind in (training_pipeline.get("stage_dataset_bindings") or []):
        if not isinstance(bind, dict) or bind.get("active") is not True:
            continue
        patched = False
        for ds in (bind.get("datasets") or []):
            if not isinstance(ds, dict):
                continue
            if _canon_path(str(ds.get("path") or "")) == active_path_c:
                ds["in_tokenizer_corpus"] = True
                ds["in_tokenizer_corpus_source"] = "roundtrip"
                patched = True
        if patched:
            ds_list = [d for d in (bind.get("datasets") or []) if isinstance(d, dict)]
            cov = bind.get("tokenizer_coverage")
            if isinstance(cov, dict):
                cov["in_corpus"] = sum(1 for d in ds_list if d.get("in_tokenizer_corpus") is True)
                cov["not_in_corpus"] = sum(1 for d in ds_list if d.get("in_tokenizer_corpus") is not True)

    # Patch pipeline.pipeline.stages (used by visualizer in strictManifestMode).
    for ps in (training_pipeline.get("pipeline", {}).get("stages") or []):
        if not isinstance(ps, dict):
            continue
        for ds in (ps.get("datasets") or []):
            if not isinstance(ds, dict):
                continue
            if _canon_path(str(ds.get("path") or "")) == active_path_c:
                ds["in_tokenizer_corpus"] = True
                ds["in_tokenizer_corpus_source"] = "roundtrip"
        # Recompute tokenizer_coverage if present.
        cov = ps.get("tokenizer_coverage")
        if isinstance(cov, dict):
            ds_list = [d for d in (ps.get("datasets") or []) if isinstance(d, dict)]
            cov["in_corpus"] = sum(1 for d in ds_list if isinstance(d.get("in_tokenizer_corpus"), bool) and d["in_tokenizer_corpus"])
            cov["not_in_corpus"] = sum(1 for d in ds_list if not (isinstance(d.get("in_tokenizer_corpus"), bool) and d["in_tokenizer_corpus"]))

    # Also fix tokenizer_dataset_coverage at the top level.
    top_cov = training_pipeline.get("tokenizer_dataset_coverage")
    if isinstance(top_cov, dict):
        top_cov["active_dataset_in_corpus"] = True
        top_cov["status"] = "pass"
        if not top_cov.get("note"):
            top_cov["note"] = lin.get("coverage_note", "roundtrip exact match")


def _run_ck_train(
    args: argparse.Namespace,
    dataset_path: Path,
    token_file: Path | None,
    ck_json: Path,
) -> None:
    run_dir = Path(args.run).expanduser().resolve()
    train_driver = str(getattr(args, "train_driver", "ck_run") or "ck_run").strip().lower()
    if train_driver == "ck_cli":
        if str(args.optimizer).lower() != "adamw":
            raise RuntimeError("ck_cli train driver currently supports optimizer=adamw only.")
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
        "--train-optimizer",
        str(args.optimizer),
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


def _run_v7_init(args: argparse.Namespace, run_dir: Path) -> None:
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


def main() -> int:
    ap = argparse.ArgumentParser(description="High-level v7 dataset/tokenizer/train pipeline")
    ap.add_argument(
        "--run",
        required=True,
        help="Existing v7 run-dir (created by ck_run_v7.py init). Recommended: ~/.cache/ck-engine-v7/models/train/<run-name> for ir_hub discovery.",
    )
    ap.add_argument(
        "--init-if-missing",
        action="store_true",
        help="Auto-run v7 init when --run is missing or missing weights_manifest.json/weights.bump",
    )
    ap.add_argument("--init", default="xavier_uniform", choices=["normal_0p02", "xavier_uniform", "xavier_normal", "kaiming_uniform", "zeros"])
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--vocab-size", type=int, default=None, help="Run vocab size for init (default: 256 byte, bpe-vocab-size for bpe/ascii_bpe)")
    ap.add_argument("--embed-dim", type=int, default=128)
    ap.add_argument("--hidden-dim", type=int, default=256)
    ap.add_argument("--num-heads", type=int, default=8)
    ap.add_argument("--num-kv-heads", type=int, default=4)
    ap.add_argument("--context-len", type=int, default=128)
    ap.add_argument("--template", default="qwen3")
    ap.add_argument(
        "--curriculum-stage",
        choices=["auto", "stage_a", "stage_b", "sft", "dpo", "grpo", "ppo"],
        default="auto",
        help="Annotate training_pipeline stage metadata for visualizer flow (stage_a=pretrain, stage_b=midtrain, sft/dpo/grpo/ppo=post-pretrain phases)",
    )
    ap.add_argument("--data", default=None, help="UTF-8 training text file path")
    ap.add_argument("--dataset-repeats", type=int, default=10, help="If --data missing, create repeated SVG rows")
    ap.add_argument("--tokenizer", choices=["byte", "bpe", "ascii_bpe"], default="byte", help="Tokenization path for training")
    ap.add_argument(
        "--pack-mode",
        choices=["stream", "sample"],
        default="stream",
        help=(
            "Token packing strategy. stream=continuous token stream windows; "
            "sample=pack complete non-empty rows into seq_len windows (no cross-row bleed)."
        ),
    )
    ap.set_defaults(pack_total_tokens_from_windows=True)
    ap.add_argument(
        "--pack-total-tokens-from-windows",
        dest="pack_total_tokens_from_windows",
        action="store_true",
        help=(
            "When --pack-mode sample, override --total-tokens with "
            "recommended_total_tokens from pack report (default: enabled)."
        ),
    )
    ap.add_argument(
        "--no-pack-total-tokens-from-windows",
        dest="pack_total_tokens_from_windows",
        action="store_false",
        help=(
            "When --pack-mode sample, keep CLI --total-tokens instead of "
            "recommended_total_tokens from pack report."
        ),
    )
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
        help="Fail if any non-empty row is not SVG-compatible (<svg...>, <task>...</task><svg...>, or [tags]<svg...>)",
    )
    ap.add_argument(
        "--spec-catalog",
        default=None,
        help="Optional spec catalog JSON associated with this dataset generation run.",
    )
    ap.add_argument(
        "--strict-coverage-gate",
        action="store_true",
        help="Fail fast if coverage manifest gate reports missing spec/tag coverage.",
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
    ap.add_argument("--optimizer", choices=["adamw", "sgd"], default="adamw")
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--enforce-production-safety", action="store_true")
    ap.add_argument(
        "--resume-latest-checkpoint",
        action="store_true",
        help="Before training, promote latest checkpoint into run_dir weights (resume-in-place).",
    )
    ap.add_argument(
        "--resume-step",
        type=int,
        default=None,
        help="Optional exact checkpoint step to resume from (implies --resume-latest-checkpoint).",
    )
    ap.add_argument("--with-torch-ref", action="store_true", help="Run torch ref too (byte/bpe/ascii_bpe via token-file)")
    ap.set_defaults(open_visualizer=True)
    ap.add_argument("--open-visualizer", dest="open_visualizer", action="store_true",
                    help="Generate v7 IR visualizer HTML after training (default: enabled)")
    ap.add_argument("--no-open-visualizer", dest="open_visualizer", action="store_false",
                    help="Skip v7 IR visualizer HTML generation")
    ap.add_argument("--json-out", default=None, help="Optional pipeline report JSON")
    ap.add_argument("--bpe-vocab-size", type=int, default=1024)
    ap.add_argument("--bpe-min-freq", type=int, default=2)
    ap.add_argument(
        "--bpe-max-piece-bytes",
        type=int,
        default=0,
        help="Max merged token piece length in bytes for BPE training (0 = unbounded).",
    )
    ap.add_argument("--bpe-threads", type=int, default=4)
    ap.add_argument(
        "--reuse-run-tokenizer",
        action="store_true",
        help=(
            "For bpe/ascii_bpe modes, reuse run_dir tokenizer.json + tokenizer_bin "
            "instead of retraining BPE. Recommended for checkpoint continuation."
        ),
    )
    ap.add_argument(
        "--allow-tokenizer-retrain-on-resume",
        action="store_true",
        help=(
            "Dangerous override: permit tokenizer retrain while resuming checkpoints. "
            "Use only when intentionally starting a new tokenizer/model line."
        ),
    )
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
    ap.add_argument(
        "--backfill-ledger",
        action="store_true",
        help="Backfill run_ledger.jsonl for existing .ck_pipeline runs that lack a ledger entry, then exit.",
    )
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
    if args.bpe_max_piece_bytes < 0:
        _errors.append(f"--bpe-max-piece-bytes must be >= 0, got {args.bpe_max_piece_bytes}")
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
    if args.resume_step is not None and int(args.resume_step) < 0:
        _errors.append(f"--resume-step must be >= 0, got {args.resume_step}")
    if _errors:
        raise SystemExit("ERROR: invalid arguments:\\n  " + "\\n  ".join(_errors))

    if bool(getattr(args, "with_torch_ref", False)) and str(args.optimizer).lower() != "adamw":
        raise SystemExit("ERROR: --with-torch-ref currently supports --optimizer adamw only.")

    if args.require_ascii_data is None:
        args.require_ascii_data = args.tokenizer == "ascii_bpe"

    run_dir = Path(args.run).expanduser().resolve()
    cache_train_root = (Path.home() / ".cache" / "ck-engine-v7" / "models" / "train").resolve()
    try:
        in_cache_train_root = run_dir.is_relative_to(cache_train_root)
    except AttributeError:
        in_cache_train_root = str(run_dir).startswith(str(cache_train_root))
    if not in_cache_train_root:
        print(
            "[WARN] --run is outside ~/.cache/ck-engine-v7/models/train.\n"
            f"       run_dir={run_dir}\n"
            "       open_ir_hub.py may not auto-discover this run unless you move it later."
        )
    manifest = run_dir / "weights_manifest.json"
    weights_bump = run_dir / "weights.bump"
    needs_init = (not run_dir.exists()) or (not manifest.exists()) or (not weights_bump.exists())
    if needs_init:
        if not args.init_if_missing:
            if not run_dir.exists():
                raise SystemExit(
                    f"ERROR: run-dir not found: {run_dir}\n"
                    "Hint: pass --init-if-missing to bootstrap automatically."
                )
            missing = []
            if not manifest.exists():
                missing.append(str(manifest))
            if not weights_bump.exists():
                missing.append(str(weights_bump))
            missing_text = "\n  - ".join(missing) if missing else "unknown"
            raise SystemExit(
                "ERROR: run-dir exists but is not initialized for training.\n"
                f"run-dir: {run_dir}\n"
                f"missing:\n  - {missing_text}\n"
                "Hint: pass --init-if-missing (or run ck_run_v7.py init first)."
            )
        if run_dir.exists() and (not manifest.exists() or not weights_bump.exists()):
            print(f"[init] run-dir exists but missing manifest/weights, bootstrapping: {run_dir}")
        _run_v7_init(args, run_dir)

    # ── --backfill-ledger: one-time migration mode ─────────────────────────
    if getattr(args, "backfill_ledger", False):
        n = _backfill_run_ledger(run_dir)
        print(f"[ledger-backfill] appended {n} record(s) to {run_dir}/run_ledger.jsonl")
        return 0

    # ── Auto-backfill: silently migrate legacy runs on every invocation ────
    _backfill_run_ledger(run_dir)

    if args.resume_step is not None:
        args.resume_latest_checkpoint = True
    if (
        bool(getattr(args, "resume_latest_checkpoint", False))
        and _is_bpe_tokenizer_mode(str(args.tokenizer))
        and not bool(getattr(args, "reuse_run_tokenizer", False))
        and not bool(getattr(args, "allow_tokenizer_retrain_on_resume", False))
    ):
        raise SystemExit(
            "ERROR: refusing tokenizer retrain during checkpoint resume.\n"
            "Use --reuse-run-tokenizer for true continuation, or pass\n"
            "--allow-tokenizer-retrain-on-resume only if you intentionally want a new tokenizer/model line."
        )
    resume_checkpoint: dict[str, Any] = {
        "status": "skipped",
        "reason": "resume_not_requested",
    }
    if bool(getattr(args, "resume_latest_checkpoint", False)):
        resume_checkpoint = _promote_checkpoint(
            run_dir,
            strict=True,
            step=(int(args.resume_step) if args.resume_step is not None else None),
            purpose="resume_before_train",
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
    coverage_gate = _check_coverage_gate(
        dataset_path=dataset_path,
        spec_catalog=(str(args.spec_catalog) if args.spec_catalog else None),
        strict=bool(args.strict_coverage_gate),
    )
    if str(coverage_gate.get("status")) != "skipped":
        print(
            "[coverage-gate] "
            f"status={coverage_gate.get('status')} "
            f"passed={coverage_gate.get('passed')} "
            f"manifest={coverage_gate.get('manifest_path')}"
        )

    ck_json = work_dir / "train_ck.json"
    torch_json = work_dir / "train_torch_ref.json"
    token_file: Path | None = None
    bpe_artifacts: dict[str, Any] = {}
    tokenizer_json_for_roundtrip: str | None = None
    token_ids_all: list[int] = []
    train_token_ids: list[int] = []
    decoded_text: str = ""
    dataset_text = dataset_path.read_text(encoding="utf-8", errors="ignore")
    encode_line_fn: Callable[[str], list[int]]
    decode_ids_fn: Callable[[list[int]], str]
    bpe_handle: _TrueBPEHandle | None = None
    pack_stats: dict[str, Any] = {"mode": str(args.pack_mode)}
    pack_report: dict[str, Any] | None = None

    if _is_bpe_tokenizer_mode(args.tokenizer):
        _ensure_binary(BPE_BIN, "ck-bpe-train")
        _ensure_binary(TOKENIZER_LIB, "tokenizer")
        run_tokenizer_json = run_dir / "tokenizer.json"
        run_tokenizer_bin = run_dir / "tokenizer_bin"
        if bool(getattr(args, "reuse_run_tokenizer", False)):
            if not run_tokenizer_json.exists():
                raise SystemExit(
                    "ERROR: --reuse-run-tokenizer requested but run_dir/tokenizer.json is missing.\n"
                    f"  run_dir: {run_dir}"
                )
            if not run_tokenizer_bin.exists():
                raise SystemExit(
                    "ERROR: --reuse-run-tokenizer requested but run_dir/tokenizer_bin is missing.\n"
                    f"  run_dir: {run_dir}"
                )
            bpe_handle = _TrueBPEHandle(TOKENIZER_LIB, run_tokenizer_bin)
            tokenizer_json = run_tokenizer_json
            bpe_bin_dir = run_tokenizer_bin
            run_bpe_bin_dir = run_tokenizer_bin
        else:
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
                "--max-piece-bytes",
                str(args.bpe_max_piece_bytes),
                "--threads",
                str(args.bpe_threads),
            ]
            if args.tokenizer == "ascii_bpe":
                bpe_cmd.append("--ascii-only")
            _run(bpe_cmd, cwd=ROOT)
            bpe_handle = _TrueBPEHandle(TOKENIZER_LIB, bpe_bin_dir)
            run_bpe_bin_dir = _sync_bpe_artifacts_to_run(run_dir, tokenizer_json, bpe_bin_dir)
        ids = _encode_large_text_with_bpe_handle(bpe_handle, dataset_text)
        if len(ids) <= 1:
            raise SystemExit("ERROR: BPE encoding produced <=1 token; provide richer data.")
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
        token_ids_all = [int(v) for v in ids]
        train_token_ids = list(token_ids_all)
        if str(args.pack_mode) == "sample":
            token_file = work_dir / "train_tokens.txt"
            pack_report_json = work_dir / "train_token_pack.json"
            token_file, pack_report, pack_stats = _run_sample_packer(
                dataset_path=dataset_path,
                tokenizer_json=run_tokenizer_json,
                tokenizer_bin=run_bpe_bin_dir,
                seq_len=int(args.seq_len),
                out_token_file=token_file,
                out_report_json=pack_report_json,
            )
            if bool(getattr(args, "pack_total_tokens_from_windows", True)):
                rec_tokens = int(pack_stats.get("recommended_total_tokens", 0) or 0)
                if rec_tokens > 0:
                    prev_tokens = int(args.total_tokens)
                    args.total_tokens = int(rec_tokens)
                    print(
                        "[pack] "
                        f"overriding total_tokens from {prev_tokens} -> {args.total_tokens} "
                        "(sample windows)"
                    )
            # Count from written token stream
            try:
                train_token_ids = [
                    int(line.strip()) for line in token_file.read_text(encoding="utf-8").splitlines() if line.strip()
                ]
            except Exception:
                train_token_ids = []
        else:
            token_file = work_dir / "train_tokens.txt"
            _atomic_write_text(token_file, "\n".join(str(v) for v in train_token_ids) + "\n")
        tokenizer_json_for_roundtrip = str(run_dir / "tokenizer.json")
        decoded_text = bpe_handle.decode(token_ids_all)
        encode_line_fn = lambda row: _encode_segment_with_bpe_fallback(
            bpe_handle, row, chunk_chars=2048
        )
        decode_ids_fn = bpe_handle.decode
        bpe_artifacts = {
            "tokenizer_json": str(tokenizer_json),
            "binary_dir": str(bpe_bin_dir),
            "run_tokenizer_json": str(run_tokenizer_json),
            "run_binary_dir": str(run_bpe_bin_dir),
            "token_file": str(token_file),
            "token_count": int(len(train_token_ids)),
            "raw_token_count": int(len(token_ids_all)),
            "mode": "ascii_bpe" if args.tokenizer == "ascii_bpe" else "bytelevel_bpe",
            "reused_run_tokenizer": bool(getattr(args, "reuse_run_tokenizer", False)),
            "packing": dict(pack_stats),
            "pack_report": pack_report,
        }

    if str(args.train_driver) == "ck_cli" and token_file is None:
        # Native ck-cli train path consumes deterministic integer token streams.
        ids = list(dataset_path.read_bytes())
        if len(ids) <= 1:
            raise SystemExit("ERROR: byte tokenizer path produced <=1 token; provide richer data.")
        train_token_ids = [int(v) for v in ids]
        if str(args.pack_mode) == "sample":
            rows = _read_non_empty_rows(dataset_path)
            row_ids_all: list[list[int]] = [
                [int(v) for v in row.encode("utf-8")] for row in rows if row.strip()
            ]
            pad_id = _resolve_pad_token_id(run_dir, default=0)
            train_token_ids, pack_stats = _pack_rows_to_seq_windows(
                row_ids_all, int(args.seq_len), int(pad_id)
            )
        token_file = work_dir / "train_tokens.txt"
        _atomic_write_text(token_file, "\n".join(str(v) for v in train_token_ids) + "\n")

    if not _is_bpe_tokenizer_mode(args.tokenizer):
        raw = dataset_path.read_bytes()
        token_ids_all = [int(v) for v in raw]
        decoded_text = raw.decode("utf-8", errors="replace")
        encode_line_fn = lambda row: [int(v) for v in row.encode("utf-8")]
        decode_ids_fn = lambda row_ids: bytes(int(v) & 0xFF for v in row_ids).decode("utf-8", errors="replace")
        if str(args.pack_mode) == "sample" and token_file is None:
            rows = _read_non_empty_rows(dataset_path)
            row_ids_all: list[list[int]] = [
                [int(v) for v in row.encode("utf-8")] for row in rows if row.strip()
            ]
            pad_id = _resolve_pad_token_id(run_dir, default=0)
            train_token_ids, pack_stats = _pack_rows_to_seq_windows(
                row_ids_all, int(args.seq_len), int(pad_id)
            )
            token_file = work_dir / "train_tokens.txt"
            _atomic_write_text(token_file, "\n".join(str(v) for v in train_token_ids) + "\n")

    if str(args.pack_mode) == "sample":
        rows_kept = (
            int(pack_stats.get("rows", 0) or 0)
            if isinstance(pack_stats, dict)
            else 0
        )
        if rows_kept <= 0 and isinstance(pack_report, dict):
            rows_kept = int(pack_report.get("rows_kept", 0) or 0)
        print(
            "[pack] "
            f"mode=sample rows={rows_kept} windows={pack_stats.get('windows', 0)} "
            f"seq_len={pack_stats.get('seq_len', args.seq_len)} fill={float(pack_stats.get('fill_ratio', 0.0)):.4f} "
            f"pad_tokens={pack_stats.get('pad_tokens', 0)}"
        )

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
    cov_manifest_path = coverage_gate.get("manifest_path")
    if isinstance(cov_manifest_path, str) and cov_manifest_path:
        data_lab_artifacts["coverage_manifest_json"] = cov_manifest_path
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
            "curriculum_stage": str(args.curriculum_stage),
            "train_driver": str(args.train_driver),
            "prepare_only": True,
            "artifacts": {
                "work_dir": str(work_dir),
                "token_file": str(token_file) if token_file is not None else None,
                "bpe": bpe_artifacts or None,
                "data_lab": data_lab_artifacts,
            },
            "dataset_qc": dataset_qc,
            "coverage_gate": coverage_gate,
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
            resume_checkpoint=resume_checkpoint,
        )
        training_plan_path = run_dir / "training_plan.json"
        existing_plan: dict[str, Any] | None = None
        if training_plan_path.exists():
            try:
                existing = _load_json(training_plan_path)
                if isinstance(existing, dict):
                    existing_plan = existing
            except Exception:
                existing_plan = None
        training_plan = _build_or_update_training_plan_payload(
            run_dir=run_dir,
            training_pipeline=prepare_pipeline,
            existing_plan=existing_plan,
        )
        _apply_training_plan_to_pipeline_payload(prepare_pipeline, training_plan)
        # Apply roundtrip coverage AFTER the plan rebuilds stage_dataset_bindings so
        # the coverage patch is not overwritten.
        _apply_roundtrip_coverage_to_pipeline(prepare_pipeline, tokenizer_roundtrip, dataset_path)
        _atomic_write_text(training_plan_path, json.dumps(training_plan, indent=2))
        prepare_pipeline["training_plan_path"] = str(training_plan_path)
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

    # ── Run Ledger: write start record + pipeline_report stub ──────────────
    # Compute active_stage from curriculum_stage (mirrors _build_training_pipeline_payload).
    _curriculum_raw = str(getattr(args, "curriculum_stage", "auto") or "auto").strip().lower()
    _stage_map = {
        "stage_b": "midtrain", "midtrain": "midtrain",
        "stage_a": "pretrain", "pretrain": "pretrain",
        "sft": "sft", "dpo": "dpo", "grpo": "grpo", "ppo": "ppo",
    }
    _active_stage_for_ledger = _stage_map.get(_curriculum_raw, "pretrain")
    _pipeline_stub = {
        "format": "v7-train-data-pipeline",
        "run_dir": str(run_dir),
        "dataset": str(dataset_path),
        "curriculum_stage": str(args.curriculum_stage),
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    _atomic_write_text(work_dir / "pipeline_report.json", json.dumps(_pipeline_stub, indent=2))
    ledger_rec = _build_ledger_start_record(run_dir, work_dir, args, _active_stage_for_ledger)
    _append_ledger_entry(run_dir, ledger_rec)
    # ── End ledger start ───────────────────────────────────────────────────

    _run_ck_train(args, dataset_path, token_file, ck_json)

    # ── Run Ledger: append completion record ───────────────────────────────
    _ck_json_data: dict[str, Any] = {}
    if ck_json.exists():
        try:
            _ck_json_data = _load_json(ck_json)
        except Exception:
            pass
    _lc = _ck_json_data.get("loss_curve") if isinstance(_ck_json_data.get("loss_curve"), list) else []
    _lc_finite = [p for p in _lc if isinstance(p, dict) and isinstance(p.get("loss_ck"), (int, float))]
    _loss_vals = [float(p["loss_ck"]) for p in _lc_finite]
    _loss_first = _loss_vals[0] if _loss_vals else None
    _loss_final = _loss_vals[-1] if _loss_vals else None
    _loss_min_idx = _loss_vals.index(min(_loss_vals)) if _loss_vals else None
    _ck_steps_value: int | None = None
    try:
        _ck_steps_value = int(_ck_json_data.get("steps"))
    except Exception:
        _ck_steps_value = None
    ckpt_snapshot = _snapshot_run_checkpoint(run_dir, work_dir, _ck_steps_value)
    _append_ledger_entry(run_dir, {
        **ledger_rec,
        "status": "completed",
        "ended_at": datetime.now(timezone.utc).isoformat(),
        "steps": _ck_json_data.get("steps"),
        "total_tokens": _ck_json_data.get("total_tokens"),
        "loss_first": _loss_first,
        "loss_final": _loss_final,
        "loss_min": min(_loss_vals) if _loss_vals else None,
        "loss_min_step": int(_loss_min_idx + 1) if _loss_min_idx is not None else None,
        "checkpoint_step": _ck_steps_value,
        "checkpoint_bump": ckpt_snapshot.get("bump"),
        "checkpoint_manifest": ckpt_snapshot.get("manifest"),
    })
    # ── End ledger completion ──────────────────────────────────────────────

    if args.with_torch_ref:
        _run_torch_ref(args, dataset_path, torch_json, token_file=token_file)

    checkpoint_promotion = _promote_latest_checkpoint_for_eval(
        run_dir,
        strict=bool(args.strict_data_gates and args.require_svg_rows),
    )

    post_train_eval, eval_model_dir = _run_post_train_svg_eval(args, run_dir)
    post_train_eval_path: str | None = None
    if isinstance(post_train_eval, dict) and post_train_eval:
        post_eval_doc = dict(post_train_eval)
        if eval_model_dir:
            post_eval_doc["model_dir"] = str(eval_model_dir)
        post_eval_doc["checkpoint_promotion"] = dict(checkpoint_promotion)
        post_eval_doc.setdefault("generated_at", datetime.now(timezone.utc).isoformat())
        post_eval_file = run_dir / "post_train_eval.json"
        _atomic_write_text(post_eval_file, json.dumps(post_eval_doc, indent=2))
        post_train_eval_path = str(post_eval_file)

    if args.strict_data_gates and bool(args.require_svg_rows):
        if not bool(args.post_train_eval):
            print(
                "WARN: strict data gates requested with --no-post-train-eval; "
                "skipping output-quality SVG gate for this run."
            )
        else:
            if post_train_eval.get("status") != "ok":
                raise SystemExit(
                    "ERROR: strict data gate failed (post-train eval unavailable).\n"
                    f"  status: {post_train_eval.get('status')}\n"
                    f"  reason: {post_train_eval.get('reason')}\n"
                    "  note: this gate is output-quality/data-fit, not CK-vs-PyTorch math parity."
                )
            valid_svg_rate = float(post_train_eval.get("valid_svg_rate", 0.0))
            if valid_svg_rate < float(args.min_valid_svg_rate):
                diagnosis = post_train_eval.get("quality_diagnosis")
                diag_text = ",".join([str(x) for x in diagnosis]) if isinstance(diagnosis, list) and diagnosis else "n/a"
                raise SystemExit(
                    "ERROR: strict data gate failed (valid SVG rate).\n"
                    f"  valid_svg_rate: {valid_svg_rate:.4f}\n"
                    f"  threshold:      {float(args.min_valid_svg_rate):.4f}\n"
                    f"  diagnosis:      {diag_text}\n"
                    f"  artifact:       {post_train_eval_path or (run_dir / 'post_train_eval.json')}\n"
                    "  note: this is a data/task quality failure, not a numerical parity failure.\n"
                    "  next: expand SVG data + add instruction-to-SVG SFT pairs, then rerun."
                )

    report = {
        "format": "v7-train-data-pipeline",
        "run_dir": str(run_dir),
        "dataset": str(dataset_path),
        "tokenizer": str(args.tokenizer),
        "curriculum_stage": str(args.curriculum_stage),
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
            "checkpoint_snapshot": ckpt_snapshot,
        },
        "dataset_qc": dataset_qc,
        "coverage_gate": coverage_gate,
        "dataset_profile": dataset_profile,
        "tokenizer_roundtrip": tokenizer_roundtrip,
        "post_train_eval": post_train_eval,
        "checkpoint_promotion": checkpoint_promotion,
        "resume_checkpoint": resume_checkpoint,
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
        resume_checkpoint=resume_checkpoint,
    )
    training_plan_path = run_dir / "training_plan.json"
    existing_plan: dict[str, Any] | None = None
    if training_plan_path.exists():
        try:
            existing = _load_json(training_plan_path)
            if isinstance(existing, dict):
                existing_plan = existing
        except Exception:
            existing_plan = None
    training_plan = _build_or_update_training_plan_payload(
        run_dir=run_dir,
        training_pipeline=training_pipeline,
        existing_plan=existing_plan,
    )
    _apply_training_plan_to_pipeline_payload(training_pipeline, training_plan)
    _apply_roundtrip_coverage_to_pipeline(training_pipeline, tokenizer_roundtrip, dataset_path)
    _atomic_write_text(training_plan_path, json.dumps(training_plan, indent=2))
    training_pipeline["training_plan_path"] = str(training_plan_path)
    pipeline_json_path = run_dir / "training_pipeline_latest.json"
    _atomic_write_text(pipeline_json_path, json.dumps(training_pipeline, indent=2))

    print("v7 train pipeline complete")
    print(f"  run_dir:   {run_dir}")
    print(f"  dataset:   {dataset_path}")
    print(f"  tokenizer: {args.tokenizer}")
    print(f"  driver:    {args.train_driver}")
    if isinstance(resume_checkpoint, dict):
        r_status = str(resume_checkpoint.get("status", "unknown"))
        r_strategy = str(resume_checkpoint.get("strategy", "-"))
        r_step = resume_checkpoint.get("step")
        if r_status == "ok":
            if r_step is None:
                print(f"  resume:    {r_status} ({r_strategy})")
            else:
                print(f"  resume:    {r_status} ({r_strategy}, step={r_step})")
    print(f"  report:    {out_path}")
    print(f"  plan:      {training_plan_path}")
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
    if isinstance(ckpt_snapshot, dict):
        if str(ckpt_snapshot.get("status")) == "ok":
            print(
                "  ckpt_snap: "
                f"step={ckpt_snapshot.get('step')} "
                f"path={ckpt_snapshot.get('bump')}"
            )
        else:
            print(
                "  ckpt_snap: "
                f"status={ckpt_snapshot.get('status')} "
                f"reason={ckpt_snapshot.get('reason')}"
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

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
import shutil
import shlex
import struct
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
CK_RUN = ROOT / "version" / "v7" / "scripts" / "ck_run_v7.py"
TORCH_REF = ROOT / "version" / "v7" / "scripts" / "train_qwen3_torch_from_run_v7.py"
OPEN_VIS = ROOT / "version" / "v7" / "tools" / "open_ir_visualizer.py"
BPE_BIN = ROOT / "build" / "ck-bpe-train"
TOKENIZER_LIB = ROOT / "build" / "libckernel_tokenizer.so"
CK_CLI_BIN = ROOT / "build" / "ck-cli-v7"

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


def _encode_with_ck_true_bpe(tokenizer_lib: Path, bin_dir: Path, text: str) -> list[int]:
    if not text:
        raise RuntimeError("BPE encoding requires non-empty training text.")

    lib = _load_true_bpe_runtime(tokenizer_lib)
    bpe = lib.ck_true_bpe_create()
    if not bpe:
        raise RuntimeError("ck_true_bpe_create failed")

    try:
        vocab_size, num_merges, offsets_arr, merges_arr, strings_buf = _load_true_bpe_binary_artifacts(bin_dir)
        rc = lib.ck_true_bpe_load_binary(
            bpe,
            vocab_size,
            offsets_arr,
            ctypes.cast(strings_buf, ctypes.c_char_p),
            num_merges,
            merges_arr,
        )
        if rc != 0:
            raise RuntimeError(f"ck_true_bpe_load_binary failed rc={rc}")

        text_bytes = text.encode("utf-8")
        max_ids = max(4096, len(text_bytes) * 8)
        out = (ctypes.c_int32 * max_ids)()
        n = int(lib.ck_true_bpe_encode(bpe, text_bytes, -1, out, max_ids))
        if n <= 1:
            raise RuntimeError("BPE encoding produced <=1 token; provide richer data.")
        return [int(out[i]) for i in range(n)]
    finally:
        lib.ck_true_bpe_free(bpe)


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
        "path": str(dataset_path),
        "total_lines": int(len(rows)),
        "non_empty_lines": int(non_empty),
        "bytes": int(len(payload)),
        "require_ascii": bool(require_ascii),
        "require_svg_rows": bool(require_svg_rows),
        "ascii_violations": 0,
        "svg_violations": 0,
    }


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
        text = dataset_path.read_text(encoding="utf-8", errors="ignore")
        ids = _encode_with_ck_true_bpe(TOKENIZER_LIB, bpe_bin_dir, text)
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

    if args.token_file_out and token_file is not None:
        token_file_out = Path(args.token_file_out).expanduser().resolve()
        token_file_out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(token_file, token_file_out)
        token_file = token_file_out
        if bpe_artifacts:
            bpe_artifacts["token_file"] = str(token_file)

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
            },
            "dataset_qc": dataset_qc,
        }
        if args.json_out:
            out_path = Path(args.json_out).expanduser().resolve()
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print("v7 train pipeline prepared")
        print(f"  run_dir:   {run_dir}")
        print(f"  dataset:   {dataset_path}")
        print(f"  tokenizer: {args.tokenizer}")
        print(f"  driver:    {args.train_driver}")
        if token_file is not None:
            print(f"  token_file:{token_file}")
        return 0

    _run_ck_train(args, dataset_path, token_file, ck_json)

    if args.with_torch_ref:
        _run_torch_ref(args, dataset_path, torch_json, token_file=token_file)

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
        },
        "dataset_qc": dataset_qc,
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
    )
    pipeline_json_path = run_dir / "training_pipeline_latest.json"
    _atomic_write_text(pipeline_json_path, json.dumps(training_pipeline, indent=2))

    print("v7 train pipeline complete")
    print(f"  run_dir:   {run_dir}")
    print(f"  dataset:   {dataset_path}")
    print(f"  tokenizer: {args.tokenizer}")
    print(f"  driver:    {args.train_driver}")
    print(f"  report:    {out_path}")
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

#!/usr/bin/env python3
"""
High-level v7 training pipeline:
dataset -> (optional BPE) -> CK training -> optional monitoring helpers.

This is a convenience wrapper over existing v7 tools.
"""

from __future__ import annotations

import argparse
import json
import math
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
CK_RUN = ROOT / "version" / "v7" / "scripts" / "ck_run_v7.py"
TORCH_REF = ROOT / "version" / "v7" / "scripts" / "train_qwen3_torch_from_run_v7.py"
OPEN_VIS = ROOT / "version" / "v7" / "tools" / "open_ir_visualizer.py"
BPE_BIN = ROOT / "build" / "ck-bpe-train"

SVG_LINE = (
    '<svg width="100" height="100" xmlns="http://www.w3.org/2000/svg">'
    '<rect x="10" y="10" width="80" height="80" fill="red" stroke="black"/></svg>'
)


def _python_exec() -> str:
    venv_py = ROOT / ".venv" / "bin" / "python"
    return str(venv_py) if venv_py.exists() else sys.executable


def _run(cmd: list[str], cwd: Path) -> None:
    print("[run]", " ".join(shlex.quote(c) for c in cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)


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


def _encode_with_hf_tokenizers(tokenizer_json: Path, text: str) -> list[int]:
    try:
        from tokenizers import Tokenizer  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "Python package `tokenizers` is required for --tokenizer bpe. "
            "Install with: .venv/bin/pip install tokenizers"
        ) from e
    tok = Tokenizer.from_file(str(tokenizer_json))
    ids = tok.encode(text).ids
    if len(ids) <= 1:
        raise RuntimeError("BPE encoding produced <=1 token; provide richer data.")
    return [int(x) for x in ids]


def _make_corpus_dir_from_dataset(dataset_path: Path, work_dir: Path) -> Path:
    corpus_dir = work_dir / "corpus"
    corpus_dir.mkdir(parents=True, exist_ok=True)
    dst = corpus_dir / dataset_path.name
    dst.write_text(dataset_path.read_text(encoding="utf-8", errors="ignore"), encoding="utf-8")
    return corpus_dir


def _run_ck_train(
    args: argparse.Namespace,
    dataset_path: Path,
    token_file: Path | None,
    ck_json: Path,
) -> None:
    py = _python_exec()
    cmd = [
        py,
        str(CK_RUN),
        "train",
        "--run",
        str(Path(args.run).expanduser().resolve()),
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
    ap.add_argument("--vocab-size", type=int, default=None, help="Run vocab size for init (default: 256 byte, bpe-vocab-size for bpe)")
    ap.add_argument("--embed-dim", type=int, default=128)
    ap.add_argument("--hidden-dim", type=int, default=256)
    ap.add_argument("--num-heads", type=int, default=8)
    ap.add_argument("--num-kv-heads", type=int, default=4)
    ap.add_argument("--context-len", type=int, default=128)
    ap.add_argument("--template", default="qwen3")
    ap.add_argument("--data", default=None, help="UTF-8 training text file path")
    ap.add_argument("--dataset-repeats", type=int, default=10, help="If --data missing, create repeated SVG rows")
    ap.add_argument("--tokenizer", choices=["byte", "bpe"], default="byte", help="Tokenization path for training")
    ap.add_argument("--work-dir", default=None, help="Optional work dir for generated artifacts")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--seq-len", type=int, default=32)
    ap.add_argument("--total-tokens", type=int, default=1024)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--enforce-production-safety", action="store_true")
    ap.add_argument("--with-torch-ref", action="store_true", help="Run torch ref too (byte or bpe via token-file)")
    ap.add_argument("--open-visualizer", action="store_true", help="Open v7 IR visualizer after training")
    ap.add_argument("--json-out", default=None, help="Optional pipeline report JSON")
    ap.add_argument("--bpe-vocab-size", type=int, default=1024)
    ap.add_argument("--bpe-min-freq", type=int, default=2)
    ap.add_argument("--bpe-threads", type=int, default=4)
    args = ap.parse_args()

    run_dir = Path(args.run).expanduser().resolve()
    if not run_dir.exists():
        if not args.init_if_missing:
            raise SystemExit(
                f"ERROR: run-dir not found: {run_dir}\n"
                "Hint: pass --init-if-missing to bootstrap automatically."
            )
        init_vocab_size = int(args.vocab_size) if args.vocab_size is not None else (
            int(args.bpe_vocab_size) if args.tokenizer == "bpe" else 256
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

    ck_json = work_dir / "train_ck.json"
    torch_json = work_dir / "train_torch_ref.json"
    token_file: Path | None = None
    bpe_artifacts: dict[str, Any] = {}

    if args.tokenizer == "bpe":
        _ensure_binary(BPE_BIN, "ck-bpe-train")
        corpus_dir = _make_corpus_dir_from_dataset(dataset_path, work_dir)
        tokenizer_json = work_dir / "tokenizer.json"
        bpe_bin_dir = work_dir / "bpe_bin"
        bpe_bin_dir.mkdir(parents=True, exist_ok=True)
        _run(
            [
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
            ],
            cwd=ROOT,
        )
        text = dataset_path.read_text(encoding="utf-8", errors="ignore")
        ids = _encode_with_hf_tokenizers(tokenizer_json, text)
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
        token_file.write_text("\n".join(str(v) for v in ids) + "\n", encoding="utf-8")
        bpe_artifacts = {
            "tokenizer_json": str(tokenizer_json),
            "binary_dir": str(bpe_bin_dir),
            "token_file": str(token_file),
            "token_count": int(len(ids)),
        }

    _run_ck_train(args, dataset_path, token_file, ck_json)

    if args.with_torch_ref:
        _run_torch_ref(args, dataset_path, torch_json, token_file=token_file)

    report = {
        "format": "v7-train-data-pipeline",
        "run_dir": str(run_dir),
        "dataset": str(dataset_path),
        "tokenizer": str(args.tokenizer),
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
            "bpe": bpe_artifacts or None,
        },
        "ck_loss": {},
        "torch_loss": {},
    }

    if ck_json.exists():
        report["ck_loss"] = _loss_stats(_load_json(ck_json))
    if torch_json.exists():
        report["torch_loss"] = _loss_stats(_load_json(torch_json))

    out_path = Path(args.json_out).expanduser().resolve() if args.json_out else (work_dir / "pipeline_report.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("v7 train pipeline complete")
    print(f"  run_dir:   {run_dir}")
    print(f"  dataset:   {dataset_path}")
    print(f"  tokenizer: {args.tokenizer}")
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

#!/usr/bin/env python3
"""
SVG overfit regression gate for v7 training.

Runs a deterministic CK-runtime training pass and a PyTorch reference pass from
identical initialized run-dirs, then checks loss-shape thresholds.
"""

from __future__ import annotations

import argparse
import json
import math
import shlex
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
CK_RUN = ROOT / "version" / "v7" / "scripts" / "ck_run_v7.py"
TORCH_REF = ROOT / "version" / "v7" / "scripts" / "train_qwen3_torch_from_run_v7.py"

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


def _write_svg_dataset(path: Path, repeats: int) -> None:
    lines = [SVG_LINE for _ in range(max(1, int(repeats)))]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _extract_ck_losses(payload: dict[str, Any]) -> list[float]:
    curve = payload.get("loss_curve")
    if not isinstance(curve, list):
        return []
    out: list[float] = []
    for row in curve:
        if not isinstance(row, dict):
            continue
        val = row.get("loss_ck", row.get("loss"))
        if isinstance(val, (int, float)):
            out.append(float(val))
    return out


def _extract_torch_ref_losses(payload: dict[str, Any]) -> list[float]:
    curve = payload.get("loss_curve")
    if not isinstance(curve, list):
        return []
    out: list[float] = []
    for row in curve:
        if not isinstance(row, dict):
            continue
        val = row.get("loss")
        if isinstance(val, (int, float)):
            out.append(float(val))
    return out


@dataclass
class LossStats:
    first: float
    final: float
    minv: float
    min_step: int
    steps: int


def _loss_stats(losses: list[float]) -> LossStats | None:
    if not losses:
        return None
    min_idx = min(range(len(losses)), key=lambda i: losses[i])
    return LossStats(
        first=float(losses[0]),
        final=float(losses[-1]),
        minv=float(losses[min_idx]),
        min_step=int(min_idx + 1),
        steps=int(len(losses)),
    )


def _finite(x: float) -> bool:
    return math.isfinite(float(x))


def main() -> int:
    ap = argparse.ArgumentParser(description="v7 SVG overfit CK-vs-PyTorch regression gate")
    ap.add_argument("--work-dir", default=None, help="Optional working directory")
    ap.add_argument("--json-out", default=None, help="Optional JSON output report")

    ap.add_argument("--dataset", default=None, help="Optional existing UTF-8 dataset file")
    ap.add_argument("--dataset-repeats", type=int, default=10, help="When --dataset is not set, repeat SVG row this many times")

    ap.add_argument("--init", default="xavier_uniform", choices=["normal_0p02", "xavier_uniform", "xavier_normal", "kaiming_uniform", "zeros"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--vocab-size", type=int, default=256)
    ap.add_argument("--embed-dim", type=int, default=128)
    ap.add_argument("--hidden-dim", type=int, default=256)
    ap.add_argument("--num-heads", type=int, default=8)
    ap.add_argument("--num-kv-heads", type=int, default=4)
    ap.add_argument("--context-len", type=int, default=128)
    ap.add_argument("--template", default="qwen3")

    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--seq-len", type=int, default=32)
    ap.add_argument("--total-tokens", type=int, default=1024)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)

    # Gate thresholds
    ap.add_argument("--first-loss-tol", type=float, default=1e-5)
    ap.add_argument("--ck-min-loss-max", type=float, default=0.12)
    ap.add_argument("--pt-min-loss-max", type=float, default=0.05)
    ap.add_argument("--ck-final-loss-max", type=float, default=0.20)
    ap.add_argument("--pt-final-loss-max", type=float, default=0.20)
    ap.add_argument("--ck-pt-min-ratio-max", type=float, default=25.0)
    args = ap.parse_args()

    if args.layers < 1:
        raise SystemExit("ERROR: --layers must be >= 1")
    if args.seq_len < 1:
        raise SystemExit("ERROR: --seq-len must be >= 1")
    if args.total_tokens < args.seq_len + 1:
        raise SystemExit("ERROR: --total-tokens must be >= --seq-len + 1")

    if args.work_dir:
        work_dir = Path(args.work_dir).resolve()
        work_dir.mkdir(parents=True, exist_ok=True)
        cleanup = None
    else:
        cleanup = tempfile.TemporaryDirectory(prefix="v7_svg_overfit_gate_")
        work_dir = Path(cleanup.name)

    dataset_path = Path(args.dataset).resolve() if args.dataset else (work_dir / "svg_train.txt")
    if args.dataset:
        if not dataset_path.exists():
            raise SystemExit(f"ERROR: dataset not found: {dataset_path}")
    else:
        _write_svg_dataset(dataset_path, args.dataset_repeats)

    run_ck = work_dir / "run_ck"
    run_pt = work_dir / "run_pt"
    ck_json = work_dir / "train_svg_ck.json"
    pt_json = work_dir / "train_svg_torch_ref.json"

    py = _python_exec()
    init_common = [
        py,
        str(CK_RUN),
        "init",
        "--init",
        str(args.init),
        "--train-seed",
        str(args.seed),
        "--layers",
        str(args.layers),
        "--vocab-size",
        str(args.vocab_size),
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
    ]

    _run(init_common + ["--run", str(run_ck)], cwd=ROOT)
    _run(init_common + ["--run", str(run_pt)], cwd=ROOT)

    _run(
        [
            py,
            str(CK_RUN),
            "train",
            "--run",
            str(run_ck),
            "--backend",
            "ck",
            "--data",
            str(dataset_path),
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
            "--enforce-production-safety",
            "--train-seed",
            str(args.seed),
            "--train-json-out",
            str(ck_json),
        ],
        cwd=ROOT,
    )

    _run(
        [
            py,
            str(TORCH_REF),
            "--run-dir",
            str(run_pt),
            "--data",
            str(dataset_path),
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
            str(pt_json),
        ],
        cwd=ROOT,
    )

    ck_payload = json.loads(ck_json.read_text(encoding="utf-8"))
    pt_payload = json.loads(pt_json.read_text(encoding="utf-8"))

    ck_losses = _extract_ck_losses(ck_payload)
    pt_losses = _extract_torch_ref_losses(pt_payload)
    ck_stats = _loss_stats(ck_losses)
    pt_stats = _loss_stats(pt_losses)

    checks: list[dict[str, Any]] = []

    checks.append(
        {
            "name": "ck_loss_curve_present",
            "passed": ck_stats is not None,
            "detail": f"len={len(ck_losses)}",
        }
    )
    checks.append(
        {
            "name": "torch_loss_curve_present",
            "passed": pt_stats is not None,
            "detail": f"len={len(pt_losses)}",
        }
    )

    if ck_stats is not None and pt_stats is not None:
        first_diff = abs(float(ck_stats.first) - float(pt_stats.first))
        ck_pt_min_ratio = float(ck_stats.minv / pt_stats.minv) if pt_stats.minv > 0.0 else float("inf")
        checks.extend(
            [
                {
                    "name": "first_loss_alignment",
                    "passed": _finite(first_diff) and first_diff <= float(args.first_loss_tol),
                    "detail": f"diff={first_diff:.6e} tol={float(args.first_loss_tol):.6e}",
                },
                {
                    "name": "ck_min_loss_bound",
                    "passed": _finite(ck_stats.minv) and ck_stats.minv <= float(args.ck_min_loss_max),
                    "detail": f"ck_min={ck_stats.minv:.6f} max={float(args.ck_min_loss_max):.6f}",
                },
                {
                    "name": "torch_min_loss_bound",
                    "passed": _finite(pt_stats.minv) and pt_stats.minv <= float(args.pt_min_loss_max),
                    "detail": f"pt_min={pt_stats.minv:.6f} max={float(args.pt_min_loss_max):.6f}",
                },
                {
                    "name": "ck_final_loss_bound",
                    "passed": _finite(ck_stats.final) and ck_stats.final <= float(args.ck_final_loss_max),
                    "detail": f"ck_final={ck_stats.final:.6f} max={float(args.ck_final_loss_max):.6f}",
                },
                {
                    "name": "torch_final_loss_bound",
                    "passed": _finite(pt_stats.final) and pt_stats.final <= float(args.pt_final_loss_max),
                    "detail": f"pt_final={pt_stats.final:.6f} max={float(args.pt_final_loss_max):.6f}",
                },
                {
                    "name": "ck_vs_torch_min_ratio",
                    "passed": _finite(ck_pt_min_ratio) and ck_pt_min_ratio <= float(args.ck_pt_min_ratio_max),
                    "detail": f"ratio={ck_pt_min_ratio:.6f} max={float(args.ck_pt_min_ratio_max):.6f}",
                },
            ]
        )

    passed = all(bool(c.get("passed", False)) for c in checks)
    failures = [c for c in checks if not bool(c.get("passed", False))]

    report = {
        "format": "v7-svg-overfit-regression",
        "passed": passed,
        "failures": failures,
        "checks": checks,
        "config": {
            "work_dir": str(work_dir),
            "dataset": str(dataset_path),
            "init": {
                "seed": int(args.seed),
                "method": str(args.init),
                "layers": int(args.layers),
                "vocab_size": int(args.vocab_size),
                "embed_dim": int(args.embed_dim),
                "hidden_dim": int(args.hidden_dim),
                "num_heads": int(args.num_heads),
                "num_kv_heads": int(args.num_kv_heads),
                "context_len": int(args.context_len),
                "template": str(args.template),
            },
            "train": {
                "epochs": int(args.epochs),
                "seq_len": int(args.seq_len),
                "total_tokens": int(args.total_tokens),
                "grad_accum": int(args.grad_accum),
                "lr": float(args.lr),
                "max_grad_norm": float(args.max_grad_norm),
            },
            "thresholds": {
                "first_loss_tol": float(args.first_loss_tol),
                "ck_min_loss_max": float(args.ck_min_loss_max),
                "pt_min_loss_max": float(args.pt_min_loss_max),
                "ck_final_loss_max": float(args.ck_final_loss_max),
                "pt_final_loss_max": float(args.pt_final_loss_max),
                "ck_pt_min_ratio_max": float(args.ck_pt_min_ratio_max),
            },
        },
        "results": {
            "ck": (None if ck_stats is None else ck_stats.__dict__),
            "torch_ref": (None if pt_stats is None else pt_stats.__dict__),
        },
        "artifacts": {
            "ck_json": str(ck_json),
            "torch_ref_json": str(pt_json),
            "run_ck": str(run_ck),
            "run_pt": str(run_pt),
        },
    }

    print("v7 SVG overfit regression")
    if ck_stats is not None:
        print(
            "  CK: first=%.6f final=%.6f min=%.6f (step=%d) steps=%d"
            % (ck_stats.first, ck_stats.final, ck_stats.minv, ck_stats.min_step, ck_stats.steps)
        )
    if pt_stats is not None:
        print(
            "  PT: first=%.6f final=%.6f min=%.6f (step=%d) steps=%d"
            % (pt_stats.first, pt_stats.final, pt_stats.minv, pt_stats.min_step, pt_stats.steps)
        )
    print("  PASS" if passed else "  FAIL")

    if args.json_out:
        out_path = Path(args.json_out).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"  JSON: {out_path}")

    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

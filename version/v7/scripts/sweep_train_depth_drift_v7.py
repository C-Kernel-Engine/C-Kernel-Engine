#!/usr/bin/env python3
"""Run v7 depth sweep (e.g., 1-layer vs 2-layer) and summarize drift/parity."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CK_RUN = ROOT / "version" / "v7" / "scripts" / "ck_run_v7.py"


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_layers(text: str) -> list[int]:
    out: list[int] = []
    for part in text.split(","):
        p = part.strip()
        if not p:
            continue
        out.append(int(p))
    if not out:
        raise ValueError("--layers must include at least one integer")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Depth sweep for v7 train parity drift diagnostics")
    ap.add_argument("--run-root", type=Path, required=True)
    ap.add_argument("--layers", type=str, default="1,2")
    ap.add_argument("--init", type=str, default="xavier_uniform")
    ap.add_argument("--vocab-size", type=int, default=1024)
    ap.add_argument("--embed-dim", type=int, default=256)
    ap.add_argument("--hidden-dim", type=int, default=1024)
    ap.add_argument("--num-heads", type=int, default=8)
    ap.add_argument("--num-kv-heads", type=int, default=4)
    ap.add_argument("--context-len", type=int, default=128)
    ap.add_argument("--train-epochs", type=int, default=3)
    ap.add_argument("--train-seq-len", type=int, default=8)
    ap.add_argument("--train-total-tokens", type=int, default=4096)
    ap.add_argument("--train-grad-accum", type=int, default=8)
    ap.add_argument("--train-lr", type=float, default=5e-4)
    ap.add_argument("--prompt", type=str, default="Hello!")
    ap.add_argument("--python", type=str, default=sys.executable)
    args = ap.parse_args()

    if not CK_RUN.exists():
        print(f"ERROR: missing script: {CK_RUN}", file=sys.stderr)
        return 2

    layer_values = _parse_layers(args.layers)
    args.run_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for layers in layer_values:
        run_dir = (args.run_root / f"layers_{layers}").resolve()
        run_dir.mkdir(parents=True, exist_ok=True)
        train_json = run_dir / "train_e2e_latest.json"

        init_cmd = [
            args.python,
            str(CK_RUN),
            "init",
            "--run",
            str(run_dir),
            "--init",
            args.init,
            "--layers",
            str(layers),
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
            "--generate-ir",
            "--generate-runtime",
            "--strict",
        ]
        _run(init_cmd)

        train_cmd = [
            args.python,
            str(CK_RUN),
            "train",
            "--run",
            str(run_dir),
            "--backend",
            "both",
            "--train-epochs",
            str(args.train_epochs),
            "--train-seq-len",
            str(args.train_seq_len),
            "--train-total-tokens",
            str(args.train_total_tokens),
            "--train-grad-accum",
            str(args.train_grad_accum),
            "--train-vocab",
            str(args.vocab_size),
            "--train-d-model",
            str(args.embed_dim),
            "--train-hidden",
            str(args.hidden_dim),
            "--train-lr",
            str(args.train_lr),
            "--prompt",
            args.prompt,
            "--profile-train",
            "none",
            "--train-json-out",
            str(train_json),
        ]
        _run(train_cmd)

        report = _load_json(train_json)
        drift = report.get("drift_diagnostics", {}) if isinstance(report, dict) else {}
        rows.append(
            {
                "layers": int(layers),
                "run_dir": str(run_dir),
                "pass_parity": bool(report.get("pass_parity", False)),
                "steps": int(report.get("steps", 0)),
                "max_loss_abs_diff": float(report.get("max_loss_abs_diff", 0.0)),
                "final_param_max_abs_diff": float(report.get("final_param_max_abs_diff", 0.0)),
                "first_loss_fail_step": drift.get("first_loss_fail_step"),
                "first_param_fail_step": drift.get("first_param_fail_step"),
                "max_logit_abs_diff": float(drift.get("max_logit_abs_diff", 0.0) or 0.0),
                "max_grad_abs_diff": float(drift.get("max_grad_abs_diff", 0.0) or 0.0),
            }
        )

    summary = {
        "schema": "ck.v7.depth_sweep.v1",
        "run_root": str(args.run_root.resolve()),
        "layers": layer_values,
        "results": rows,
    }
    out = args.run_root / "depth_sweep_summary.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("layers | pass | steps | max_loss_diff | first_loss_fail | first_param_fail")
    for r in rows:
        print(
            f"{r['layers']:>6} | {str(r['pass_parity']):>4} | {r['steps']:>5} | "
            f"{r['max_loss_abs_diff']:.3e} | {str(r['first_loss_fail_step']):>15} | {str(r['first_param_fail_step']):>16}"
        )
    print(f"summary: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

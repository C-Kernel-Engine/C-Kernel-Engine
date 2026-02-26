#!/usr/bin/env python3
"""
check_runtime_replay_accum_v7.py

Why this script exists (F1 in regimen):
- Generated-runtime replay correctness under grad_accum>1.
- Verifies replay restores full train state, including accumulation snapshots,
  not just scalar loss.

This catches:
- Replay restore gaps in optimizer/accum state.
- Hidden state divergence that only appears with accumulation windows.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory


SCRIPT_DIR = Path(__file__).resolve().parent
CK_RUN = SCRIPT_DIR / "ck_run_v7.py"


def _run(cmd: list[str]) -> tuple[int, str]:
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return int(proc.returncode), str(proc.stdout)


def _bool(v: object) -> bool:
    return bool(v)


def main() -> int:
    ap = argparse.ArgumentParser(description="Runtime replay parity check for grad_accum>1 (including accum snapshots).")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--seq-len", type=int, default=8)
    ap.add_argument("--total-tokens", type=int, default=72)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--vocab", type=int, default=256)
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--prompt", type=str, default="Hello!")
    ap.add_argument("--parity-every", type=int, default=1)
    ap.add_argument("--replay-tol", type=float, default=1e-7)
    ap.add_argument("--state-tol", type=float, default=3e-5)
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args()

    if int(args.grad_accum) <= 1:
        print("ERROR: --grad-accum must be >1 for accum replay validation", file=sys.stderr)
        return 2

    with TemporaryDirectory(prefix="v7_replay_accum_") as td:
        tdir = Path(td)
        run_dir = tdir / "run"
        out_json = tdir / "train_runtime.json"

        init_cmd = [
            sys.executable,
            str(CK_RUN),
            "init",
            "--run",
            str(run_dir),
            "--train-seed",
            str(args.seed),
            "--layers",
            str(args.layers),
            "--vocab-size",
            str(args.vocab),
            "--embed-dim",
            str(args.d_model),
            "--hidden-dim",
            str(args.hidden),
            "--context-len",
            str(args.seq_len),
        ]
        init_rc, init_out = _run(init_cmd)
        if init_rc != 0:
            print(init_out, file=sys.stderr)
            print("ERROR: ck_run_v7.py init failed", file=sys.stderr)
            return 1

        train_cmd = [
            sys.executable,
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
            "adamw",
            "--train-lr",
            str(args.lr),
            "--train-max-grad-norm",
            "0",
            "--allow-unsafe-adamw-lr",
            "--enforce-production-safety",
            "--train-unsafe-adamw-lr-threshold",
            "1e-3",
            "--train-seed",
            str(args.seed),
            "--train-vocab",
            str(args.vocab),
            "--train-d-model",
            str(args.d_model),
            "--train-hidden",
            str(args.hidden),
            "--prompt",
            str(args.prompt),
            "--parity-on",
            "--parity-every",
            str(args.parity_every),
            "--parity-replay-on-check",
            "--parity-replay-tol",
            str(args.replay_tol),
            "--train-param-tol",
            str(args.state_tol),
            "--train-json-out",
            str(out_json),
        ]
        train_rc, train_out = _run(train_cmd)
        if train_rc != 0:
            print(train_out, file=sys.stderr)
            print("ERROR: ck_run_v7.py train failed", file=sys.stderr)
            return 1

        report = json.loads(out_json.read_text(encoding="utf-8"))

    oracle = report.get("oracle", {}) if isinstance(report, dict) else {}
    replay_failures = oracle.get("replay_failures", []) if isinstance(oracle, dict) else []
    parity_steps = report.get("parity_steps", []) if isinstance(report, dict) else []

    checked_steps = [row for row in parity_steps if isinstance(row, dict) and _bool(row.get("checked"))]
    checked_with_replay = [row for row in checked_steps if row.get("replay_diff") is not None]
    checked_with_accum = [
        row
        for row in checked_with_replay
        if row.get("replay_accum_snapshot_max_abs_diff") is not None
    ]

    max_replay_diff = max((float(row.get("replay_diff", 0.0) or 0.0) for row in checked_with_replay), default=0.0)
    max_accum_diff = max((float(row.get("replay_accum_snapshot_max_abs_diff", 0.0) or 0.0) for row in checked_with_accum), default=0.0)

    passed = True
    if not _bool(report.get("pass_parity", False)):
        passed = False
    if not _bool(oracle.get("replay_on_check", False)):
        passed = False
    if not _bool(oracle.get("accum_snapshot_api_available", False)):
        passed = False
    if not _bool(oracle.get("replay_accum_snapshot_api_available", False)):
        passed = False
    if len(replay_failures) > 0:
        passed = False
    if len(checked_with_replay) == 0:
        passed = False
    if len(checked_with_accum) == 0:
        passed = False
    if max_replay_diff > float(args.replay_tol):
        passed = False
    if max_accum_diff > float(args.state_tol):
        passed = False

    payload = {
        "passed": bool(passed),
        "checks": {
            "pass_parity": bool(report.get("pass_parity", False)),
            "replay_on_check": bool(oracle.get("replay_on_check", False)),
            "accum_snapshot_api_available": bool(oracle.get("accum_snapshot_api_available", False)),
            "replay_accum_snapshot_api_available": bool(oracle.get("replay_accum_snapshot_api_available", False)),
            "replay_failures_count": int(len(replay_failures)),
            "checked_steps": int(len(checked_steps)),
            "checked_steps_with_replay": int(len(checked_with_replay)),
            "checked_steps_with_accum_snapshot": int(len(checked_with_accum)),
            "max_replay_loss_abs_diff": float(max_replay_diff),
            "max_replay_accum_snapshot_abs_diff": float(max_accum_diff),
        },
        "thresholds": {
            "replay_tol": float(args.replay_tol),
            "state_tol": float(args.state_tol),
        },
        "config": {
            "epochs": int(args.epochs),
            "seq_len": int(args.seq_len),
            "total_tokens": int(args.total_tokens),
            "grad_accum": int(args.grad_accum),
            "lr": float(args.lr),
            "seed": int(args.seed),
            "vocab": int(args.vocab),
            "d_model": int(args.d_model),
            "hidden": int(args.hidden),
            "layers": int(args.layers),
            "prompt": str(args.prompt),
            "parity_every": int(args.parity_every),
        },
        "runtime_report": report,
    }

    print("=" * 88)
    print("v7 RUNTIME REPLAY ACCUM CHECK")
    print("=" * 88)
    print(f"- replay_failures_count: {payload['checks']['replay_failures_count']}")
    print(f"- checked_steps_with_replay: {payload['checks']['checked_steps_with_replay']}")
    print(f"- checked_steps_with_accum_snapshot: {payload['checks']['checked_steps_with_accum_snapshot']}")
    print(f"- max_replay_loss_abs_diff: {payload['checks']['max_replay_loss_abs_diff']:.3e}")
    print(f"- max_replay_accum_snapshot_abs_diff: {payload['checks']['max_replay_accum_snapshot_abs_diff']:.3e}")
    print("REPLAY_ACCUM:", "PASS" if passed else "FAIL")
    print("=" * 88)

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print("JSON:", args.json_out)

    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

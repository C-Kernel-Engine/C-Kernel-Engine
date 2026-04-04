#!/usr/bin/env python3
"""
check_replay_determinism_v7.py

Why this script exists (E1 in regimen):
- Determinism guardrail for the parity harness path.
- Runs identical harness config twice and asserts reproducible outputs
  (loss/param drift metrics) across repeats.

This catches:
- Seed/control-flow nondeterminism.
- Unintended order/state drift that can hide real kernel issues.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory


SCRIPT_DIR = Path(__file__).resolve().parent
TRAIN_PARITY = SCRIPT_DIR / "train_parity_epochs_v7.py"


def _default_train_root() -> Path:
    env = os.environ.get("CK_CACHE_DIR")
    if env:
        base = Path(env).expanduser()
        if base.name == "train":
            return base
        if base.name == "models":
            return base / "train"
        return base / "models" / "train"
    return Path.home() / ".cache" / "ck-engine-v7" / "models" / "train"


def _run_once(args, json_out: Path) -> dict:
    cmd = [
        sys.executable,
        str(TRAIN_PARITY),
        "--epochs",
        str(args.epochs),
        "--seq-len",
        str(args.seq_len),
        "--total-tokens",
        str(args.total_tokens),
        "--vocab",
        str(args.vocab),
        "--d-model",
        str(args.d_model),
        "--hidden",
        str(args.hidden),
        "--eps",
        str(args.eps),
        "--grad-accum",
        str(args.grad_accum),
        "--optimizer",
        str(args.optimizer),
        "--lr",
        str(args.lr),
        "--seed",
        str(args.seed),
        "--json-out",
        str(json_out),
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0:
        raise RuntimeError("train_parity run failed:\n%s" % proc.stdout)
    return json.loads(json_out.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Deterministic replay check for v7 tiny training parity.")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--total-tokens", type=int, default=1024)
    parser.add_argument("--vocab", type=int, default=256)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--eps", type=float, default=1e-5)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "sgd"])
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tol", type=float, default=1e-12)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    train_root = _default_train_root()
    train_root.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="v7_replay_", dir=str(train_root)) as td:
        tdir = Path(td)
        r1 = _run_once(args, tdir / "run1.json")
        r2 = _run_once(args, tdir / "run2.json")

    keys = [
        "max_loss_abs_diff",
        "mean_loss_abs_diff",
        "final_ck_loss",
        "final_torch_loss",
        "final_param_max_abs_diff",
        "final_param_mean_abs_diff",
    ]
    diffs = {}
    passed = bool(r1.get("pass_parity") and r2.get("pass_parity"))
    tol = float(args.tol)
    for key in keys:
        d = abs(float(r1.get(key, 0.0)) - float(r2.get(key, 0.0)))
        diffs[key] = d
        if d > tol:
            passed = False

    out = {
        "passed": passed,
        "tolerance": tol,
        "run1": r1,
        "run2": r2,
        "abs_diffs": diffs,
        "config": {
            "epochs": args.epochs,
            "seq_len": args.seq_len,
            "total_tokens": args.total_tokens,
            "vocab": args.vocab,
            "d_model": args.d_model,
            "hidden": args.hidden,
            "eps": args.eps,
            "grad_accum": args.grad_accum,
            "optimizer": args.optimizer,
            "lr": args.lr,
            "seed": args.seed,
        },
    }

    print("=" * 88)
    print("v7 REPLAY DETERMINISM CHECK")
    print("=" * 88)
    for key in keys:
        print("- %-24s diff=%.3e" % (key, out["abs_diffs"][key]))
    print("REPLAY:", "PASS" if passed else "FAIL")
    print("=" * 88)

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print("JSON:", args.json_out)

    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

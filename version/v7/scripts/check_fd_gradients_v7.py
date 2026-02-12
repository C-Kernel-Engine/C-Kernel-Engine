#!/usr/bin/env python3
"""
check_fd_gradients_v7.py

Finite-difference gradient sanity check for the v7 tiny training stack.
Compares analytical gradients (autograd + C-kernel wrappers) against
numerical central-difference estimates for selected parameters.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from train_parity_epochs_v7 import (  # noqa: E402
    TinyCKModel,
    _build_batches,
    _load_lib,
    _seed_all,
    c_cross_entropy,
)


def _get_tensor_item(param: torch.Tensor, index: tuple[int, ...]) -> float:
    ref = param
    for idx in index:
        ref = ref[idx]
    return float(ref.item())


def _set_tensor_item(param: torch.Tensor, index: tuple[int, ...], value: float) -> None:
    ref = param
    for idx in index[:-1]:
        ref = ref[idx]
    ref[index[-1]] = value


def _loss_on_batch(model: TinyCKModel, x: torch.Tensor, y: torch.Tensor, lib) -> torch.Tensor:
    logits = model(x)
    targets = y.reshape(-1)
    return c_cross_entropy(logits, targets, lib)


def run_fd_check(
    seed: int,
    seq_len: int,
    total_tokens: int,
    vocab: int,
    d_model: int,
    hidden: int,
    eps: float,
    fd_eps: float,
    tol: float,
) -> dict:
    _seed_all(seed)
    lib = _load_lib()

    model = TinyCKModel(vocab=vocab, d_model=d_model, hidden=hidden, eps=eps, lib=lib)
    model.zero_grad(set_to_none=True)

    batches = _build_batches(total_tokens=total_tokens, seq_len=seq_len, vocab=vocab, seed=seed + 7)
    if not batches:
        raise RuntimeError("No batches generated for FD check")
    x, y = batches[0]

    # Analytical gradients.
    loss = _loss_on_batch(model, x, y, lib)
    loss.backward()

    checks = [
        ("rms_gamma", (0,)),
        ("fc1.weight", (0, 0)),
        ("fc1.bias", (0,)),
        ("fc2.weight", (0, 0)),
        ("fc2.bias", (0,)),
        ("embedding.weight", (0, 0)),
    ]

    param_by_name = dict(model.named_parameters())
    results = []
    max_abs_err = 0.0
    all_pass = True

    for pname, pindex in checks:
        p = param_by_name[pname]
        analytic = _get_tensor_item(p.grad, pindex)
        orig = _get_tensor_item(p.data, pindex)

        with torch.no_grad():
            _set_tensor_item(p.data, pindex, orig + fd_eps)
        loss_plus = float(_loss_on_batch(model, x, y, lib).item())

        with torch.no_grad():
            _set_tensor_item(p.data, pindex, orig - fd_eps)
        loss_minus = float(_loss_on_batch(model, x, y, lib).item())

        with torch.no_grad():
            _set_tensor_item(p.data, pindex, orig)

        numeric = (loss_plus - loss_minus) / (2.0 * fd_eps)
        abs_err = abs(analytic - numeric)
        rel_err = abs_err / (abs(numeric) + 1e-12)
        passed = bool(abs_err <= tol)

        max_abs_err = max(max_abs_err, abs_err)
        all_pass = all_pass and passed
        results.append(
            {
                "param": pname,
                "index": list(pindex),
                "analytic": analytic,
                "numeric": numeric,
                "abs_err": abs_err,
                "rel_err": rel_err,
                "passed": passed,
            }
        )

    return {
        "passed": all_pass,
        "max_abs_err": max_abs_err,
        "tolerance": tol,
        "fd_eps": fd_eps,
        "loss": float(loss.item()),
        "checks": results,
        "config": {
            "seed": seed,
            "seq_len": seq_len,
            "total_tokens": total_tokens,
            "vocab": vocab,
            "d_model": d_model,
            "hidden": hidden,
            "eps": eps,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Finite-difference gradient check for v7 tiny stack.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--total-tokens", type=int, default=512)
    parser.add_argument("--vocab", type=int, default=256)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--eps", type=float, default=1e-5)
    parser.add_argument("--fd-eps", type=float, default=1e-3)
    parser.add_argument("--tol", type=float, default=2e-3)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    if args.seq_len <= 0 or args.total_tokens < args.seq_len + 1:
        print("ERROR: need total_tokens >= seq_len + 1", file=sys.stderr)
        return 2
    if args.fd_eps <= 0:
        print("ERROR: --fd-eps must be > 0", file=sys.stderr)
        return 2

    try:
        out = run_fd_check(
            seed=int(args.seed),
            seq_len=int(args.seq_len),
            total_tokens=int(args.total_tokens),
            vocab=int(args.vocab),
            d_model=int(args.d_model),
            hidden=int(args.hidden),
            eps=float(args.eps),
            fd_eps=float(args.fd_eps),
            tol=float(args.tol),
        )
    except Exception as exc:
        print("ERROR:", exc, file=sys.stderr)
        return 1

    print("=" * 88)
    print("v7 FINITE-DIFFERENCE GRADIENT CHECK")
    print("=" * 88)
    print("loss=%.6f fd_eps=%.2e tol=%.2e max_abs_err=%.3e" % (
        out["loss"], out["fd_eps"], out["tolerance"], out["max_abs_err"]
    ))
    for row in out["checks"]:
        print(
            "- %-18s idx=%s abs_err=%.3e rel_err=%.3e [%s]" % (
                row["param"],
                row["index"],
                row["abs_err"],
                row["rel_err"],
                "PASS" if row["passed"] else "FAIL",
            )
        )
    print("FD:", "PASS" if out["passed"] else "FAIL")
    print("=" * 88)

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print("JSON:", args.json_out)

    return 0 if out["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
v7 tiny training parity (C-kernel autograd wrappers vs pure PyTorch).

Purpose:
- Validate gradient accumulation semantics
- Validate multi-epoch optimizer behavior
- Compare parameter trajectories against PyTorch baseline

Scope:
- CPU, fp32, deterministic seed
- Tiny language-model-like stack:
  Embedding -> RMSNorm -> Linear(2H) -> SwiGLU -> Linear(V) -> CE
"""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[3]


def _load_lib() -> ctypes.CDLL:
    lib_path = ROOT / "build" / "libckernel_engine.so"
    if not lib_path.exists():
        raise FileNotFoundError(f"Missing shared library: {lib_path} (run `make` first)")
    lib = ctypes.CDLL(str(lib_path))

    lib.rmsnorm_forward.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_float,
    ]
    lib.rmsnorm_forward.restype = None

    lib.rmsnorm_backward.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    ]
    lib.rmsnorm_backward.restype = None

    lib.swiglu_forward_exact.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_int,
    ]
    lib.swiglu_forward_exact.restype = None

    lib.swiglu_backward_exact.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_int,
    ]
    lib.swiglu_backward_exact.restype = None

    lib.softmax_cross_entropy_loss.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
    ]
    lib.softmax_cross_entropy_loss.restype = None

    try:
        lib.ck_set_strict_parity.argtypes = [ctypes.c_int]
        lib.ck_set_strict_parity.restype = None
        lib.ck_set_strict_parity(1)
    except Exception:
        pass

    return lib


def _float_ptr(a: np.ndarray) -> ctypes.POINTER(ctypes.c_float):
    return a.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


def _int32_ptr(a: np.ndarray) -> ctypes.POINTER(ctypes.c_int32):
    return a.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))


class CRMSNormFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, gamma: torch.Tensor, eps: float, lib: ctypes.CDLL):
        if x.device.type != "cpu" or gamma.device.type != "cpu":
            raise RuntimeError("CRMSNormFn supports CPU tensors only")
        x_np = x.detach().contiguous().numpy().astype(np.float32, copy=False)
        gamma_np = gamma.detach().contiguous().numpy().astype(np.float32, copy=False)
        n, d = x_np.shape

        out_np = np.empty_like(x_np, dtype=np.float32)
        rstd_np = np.empty((n,), dtype=np.float32)

        lib.rmsnorm_forward(
            _float_ptr(x_np),
            _float_ptr(gamma_np),
            _float_ptr(out_np),
            _float_ptr(rstd_np),
            ctypes.c_int(n),
            ctypes.c_int(d),
            ctypes.c_int(d),
            ctypes.c_float(float(eps)),
        )

        ctx.lib = lib
        ctx.n = n
        ctx.d = d
        ctx.x_np = x_np.copy()
        ctx.gamma_np = gamma_np.copy()
        ctx.rstd_np = rstd_np.copy()
        return torch.from_numpy(out_np)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        grad_np = grad_out.detach().contiguous().numpy().astype(np.float32, copy=False)
        dx_np = np.empty_like(ctx.x_np, dtype=np.float32)
        dgamma_np = np.empty_like(ctx.gamma_np, dtype=np.float32)

        ctx.lib.rmsnorm_backward(
            _float_ptr(grad_np),
            _float_ptr(ctx.x_np),
            _float_ptr(ctx.gamma_np),
            _float_ptr(ctx.rstd_np),
            _float_ptr(dx_np),
            _float_ptr(dgamma_np),
            ctypes.c_int(ctx.n),
            ctypes.c_int(ctx.d),
            ctypes.c_int(ctx.d),
        )

        dx = torch.from_numpy(dx_np)
        dgamma = torch.from_numpy(dgamma_np)
        return dx, dgamma, None, None


class CSwiGLUFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, lib: ctypes.CDLL):
        if x.device.type != "cpu":
            raise RuntimeError("CSwiGLUFn supports CPU tensors only")
        x_np = x.detach().contiguous().numpy().astype(np.float32, copy=False)
        n, two_h = x_np.shape
        if two_h % 2 != 0:
            raise ValueError("SwiGLU input last dim must be even")
        h = two_h // 2

        out_np = np.empty((n, h), dtype=np.float32)
        lib.swiglu_forward_exact(
            _float_ptr(x_np),
            _float_ptr(out_np),
            ctypes.c_int(n),
            ctypes.c_int(h),
        )

        ctx.lib = lib
        ctx.n = n
        ctx.h = h
        ctx.x_np = x_np.copy()
        return torch.from_numpy(out_np)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        grad_np = grad_out.detach().contiguous().numpy().astype(np.float32, copy=False)
        dx_np = np.empty_like(ctx.x_np, dtype=np.float32)
        ctx.lib.swiglu_backward_exact(
            _float_ptr(ctx.x_np),
            _float_ptr(grad_np),
            _float_ptr(dx_np),
            ctypes.c_int(ctx.n),
            ctypes.c_int(ctx.h),
        )
        return torch.from_numpy(dx_np), None


class CCrossEntropyFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits: torch.Tensor, targets: torch.Tensor, lib: ctypes.CDLL):
        if logits.device.type != "cpu" or targets.device.type != "cpu":
            raise RuntimeError("CCrossEntropyFn supports CPU tensors only")
        logits_np = logits.detach().contiguous().numpy().astype(np.float32, copy=False)
        targets_np = targets.detach().contiguous().numpy().astype(np.int32, copy=False)

        n, v = logits_np.shape
        dlogits_np = np.empty_like(logits_np, dtype=np.float32)
        loss_c = ctypes.c_float(0.0)

        lib.softmax_cross_entropy_loss(
            _float_ptr(logits_np),
            _int32_ptr(targets_np),
            ctypes.c_int(n),
            ctypes.c_int(v),
            _float_ptr(dlogits_np),
            ctypes.byref(loss_c),
        )

        ctx.dlogits = torch.from_numpy(dlogits_np)
        return torch.tensor(loss_c.value, dtype=torch.float32)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        # dL/dlogits was already produced for mean CE; scale by upstream scalar.
        return ctx.dlogits * grad_out, None, None


def c_rmsnorm(x: torch.Tensor, gamma: torch.Tensor, eps: float, lib: ctypes.CDLL) -> torch.Tensor:
    return CRMSNormFn.apply(x, gamma, eps, lib)


def c_swiglu(x: torch.Tensor, lib: ctypes.CDLL) -> torch.Tensor:
    return CSwiGLUFn.apply(x, lib)


def c_cross_entropy(logits: torch.Tensor, targets: torch.Tensor, lib: ctypes.CDLL) -> torch.Tensor:
    return CCrossEntropyFn.apply(logits, targets, lib)


class TinyCKModel(nn.Module):
    def __init__(self, vocab: int, d_model: int, hidden: int, eps: float, lib: ctypes.CDLL):
        super().__init__()
        self.vocab = vocab
        self.d_model = d_model
        self.hidden = hidden
        self.eps = eps
        self.lib = lib

        self.embedding = nn.Embedding(vocab, d_model)
        self.rms_gamma = nn.Parameter(torch.ones(d_model, dtype=torch.float32))
        self.fc1 = nn.Linear(d_model, 2 * hidden)
        self.fc2 = nn.Linear(hidden, vocab)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # input_ids: [B, T] -> logits: [B*T, V]
        x = self.embedding(input_ids).reshape(-1, self.d_model)
        x = c_rmsnorm(x, self.rms_gamma, self.eps, self.lib)
        x = self.fc1(x)
        x = c_swiglu(x, self.lib)
        logits = self.fc2(x)
        return logits


class TinyTorchModel(nn.Module):
    def __init__(self, vocab: int, d_model: int, hidden: int, eps: float):
        super().__init__()
        self.vocab = vocab
        self.d_model = d_model
        self.hidden = hidden
        self.eps = eps

        self.embedding = nn.Embedding(vocab, d_model)
        self.rms_gamma = nn.Parameter(torch.ones(d_model, dtype=torch.float32))
        self.fc1 = nn.Linear(d_model, 2 * hidden)
        self.fc2 = nn.Linear(hidden, vocab)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embedding(input_ids).reshape(-1, self.d_model)
        var = x.pow(2).mean(dim=-1, keepdim=True)
        rstd = (var + self.eps).rsqrt()
        x = x * rstd * self.rms_gamma
        x = self.fc1(x)
        gate, up = x[:, : self.hidden], x[:, self.hidden :]
        x = F.silu(gate) * up
        logits = self.fc2(x)
        return logits


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _build_batches(total_tokens: int, seq_len: int, vocab: int, seed: int) -> list[Tuple[torch.Tensor, torch.Tensor]]:
    rng = np.random.default_rng(seed)
    stream = rng.integers(0, vocab, size=(total_tokens + 1,), dtype=np.int64)
    batches: list[Tuple[torch.Tensor, torch.Tensor]] = []
    for i in range(0, total_tokens - seq_len + 1, seq_len):
        x = torch.from_numpy(stream[i : i + seq_len]).long().view(1, seq_len)
        y = torch.from_numpy(stream[i + 1 : i + seq_len + 1]).long().view(1, seq_len)
        batches.append((x, y))
    return batches


def _make_optimizer(name: str, params: Iterable[torch.nn.Parameter], lr: float):
    name = name.lower()
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr)
    raise ValueError(f"Unsupported optimizer: {name}")


@dataclass
class RunStats:
    epochs: int
    seq_len: int
    total_tokens: int
    grad_accum: int
    optimizer: str
    lr: float
    steps: int
    micro_steps: int
    tokens_per_update: int
    max_loss_abs_diff: float
    mean_loss_abs_diff: float
    final_ck_loss: float
    final_torch_loss: float
    final_param_max_abs_diff: float
    final_param_mean_abs_diff: float
    pass_parity: bool


def run_training_parity(
    lib: ctypes.CDLL,
    epochs: int,
    seq_len: int,
    total_tokens: int,
    vocab: int,
    d_model: int,
    hidden: int,
    eps: float,
    grad_accum: int,
    optimizer: str,
    lr: float,
    seed: int,
    loss_tol: float,
    param_tol: float,
) -> RunStats:
    _seed_all(seed)

    model_ck = TinyCKModel(vocab=vocab, d_model=d_model, hidden=hidden, eps=eps, lib=lib)
    model_torch = TinyTorchModel(vocab=vocab, d_model=d_model, hidden=hidden, eps=eps)
    model_torch.load_state_dict(model_ck.state_dict(), strict=True)

    opt_ck = _make_optimizer(optimizer, model_ck.parameters(), lr=lr)
    opt_torch = _make_optimizer(optimizer, model_torch.parameters(), lr=lr)

    batches = _build_batches(total_tokens=total_tokens, seq_len=seq_len, vocab=vocab, seed=seed + 1)
    if not batches:
        raise RuntimeError("No batches generated; increase --total-tokens or reduce --seq-len")

    loss_diffs: list[float] = []
    last_ck = math.nan
    last_t = math.nan
    micro_count = 0
    step_count = 0

    opt_ck.zero_grad(set_to_none=True)
    opt_torch.zero_grad(set_to_none=True)

    for _epoch in range(epochs):
        for x, y in batches:
            targets = y.reshape(-1)

            logits_ck = model_ck(x)
            loss_ck = c_cross_entropy(logits_ck, targets, lib)

            logits_t = model_torch(x)
            loss_t = F.cross_entropy(logits_t, targets, reduction="mean")

            last_ck = float(loss_ck.item())
            last_t = float(loss_t.item())
            loss_diffs.append(abs(last_ck - last_t))

            (loss_ck / grad_accum).backward()
            (loss_t / grad_accum).backward()
            micro_count += 1

            if micro_count % grad_accum == 0:
                opt_ck.step()
                opt_torch.step()
                opt_ck.zero_grad(set_to_none=True)
                opt_torch.zero_grad(set_to_none=True)
                step_count += 1

    if micro_count % grad_accum != 0:
        opt_ck.step()
        opt_torch.step()
        opt_ck.zero_grad(set_to_none=True)
        opt_torch.zero_grad(set_to_none=True)
        step_count += 1

    param_diffs: list[float] = []
    for (name_ck, p_ck), (name_t, p_t) in zip(model_ck.state_dict().items(), model_torch.state_dict().items()):
        if name_ck != name_t:
            raise RuntimeError(f"State dict mismatch: {name_ck} vs {name_t}")
        diff = (p_ck - p_t).abs()
        param_diffs.append(float(diff.max().item()))

    max_loss_abs = max(loss_diffs) if loss_diffs else 0.0
    mean_loss_abs = sum(loss_diffs) / len(loss_diffs) if loss_diffs else 0.0
    max_param_abs = max(param_diffs) if param_diffs else 0.0
    mean_param_abs = sum(param_diffs) / len(param_diffs) if param_diffs else 0.0

    pass_parity = (max_loss_abs <= loss_tol) and (max_param_abs <= param_tol)

    return RunStats(
        epochs=epochs,
        seq_len=seq_len,
        total_tokens=total_tokens,
        grad_accum=grad_accum,
        optimizer=optimizer,
        lr=lr,
        steps=step_count,
        micro_steps=micro_count,
        tokens_per_update=seq_len * grad_accum,
        max_loss_abs_diff=max_loss_abs,
        mean_loss_abs_diff=mean_loss_abs,
        final_ck_loss=last_ck,
        final_torch_loss=last_t,
        final_param_max_abs_diff=max_param_abs,
        final_param_mean_abs_diff=mean_param_abs,
        pass_parity=pass_parity,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="v7 multi-epoch tiny training parity (C vs PyTorch).")
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
    parser.add_argument("--loss-tol", type=float, default=2e-5)
    parser.add_argument("--param-tol", type=float, default=2e-5)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    if args.epochs < 1:
        print("ERROR: --epochs must be >= 1", file=sys.stderr)
        return 2
    if args.grad_accum < 1:
        print("ERROR: --grad-accum must be >= 1", file=sys.stderr)
        return 2
    if args.seq_len < 1 or args.total_tokens < args.seq_len + 1:
        print("ERROR: need total_tokens >= seq_len + 1", file=sys.stderr)
        return 2

    lib = _load_lib()
    stats = run_training_parity(
        lib=lib,
        epochs=args.epochs,
        seq_len=args.seq_len,
        total_tokens=args.total_tokens,
        vocab=args.vocab,
        d_model=args.d_model,
        hidden=args.hidden,
        eps=args.eps,
        grad_accum=args.grad_accum,
        optimizer=args.optimizer,
        lr=args.lr,
        seed=args.seed,
        loss_tol=args.loss_tol,
        param_tol=args.param_tol,
    )

    print("=" * 100)
    print("v7 TRAIN PARITY (multi-epoch)")
    print("=" * 100)
    print(f"epochs={stats.epochs} seq_len={stats.seq_len} total_tokens={stats.total_tokens} "
          f"grad_accum={stats.grad_accum} optimizer={stats.optimizer} lr={stats.lr}")
    print(f"micro_steps={stats.micro_steps} optimizer_steps={stats.steps} tokens_per_update={stats.tokens_per_update}")
    print(f"max_loss_abs_diff={stats.max_loss_abs_diff:.3e} mean_loss_abs_diff={stats.mean_loss_abs_diff:.3e}")
    print(f"final_ck_loss={stats.final_ck_loss:.6f} final_torch_loss={stats.final_torch_loss:.6f}")
    print(f"final_param_max_abs_diff={stats.final_param_max_abs_diff:.3e} "
          f"final_param_mean_abs_diff={stats.final_param_mean_abs_diff:.3e}")
    print("PARITY:", "PASS" if stats.pass_parity else "FAIL")
    print("=" * 100)

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(stats.__dict__, indent=2), encoding="utf-8")
        print(f"JSON: {args.json_out}")

    return 0 if stats.pass_parity else 1


if __name__ == "__main__":
    raise SystemExit(main())

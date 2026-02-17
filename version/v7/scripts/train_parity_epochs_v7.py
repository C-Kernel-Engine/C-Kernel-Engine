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
import copy
import ctypes
import json
import math
import os
import random
import sys
import time
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
        lib.softmax_cross_entropy_loss_ptref.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
        ]
        lib.softmax_cross_entropy_loss_ptref.restype = None
    except Exception:
        pass

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
    # Custom autograd bridge: forward/backward math is executed by C kernels,
    # while PyTorch drives graph orchestration for parity validation.
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
    def forward(ctx, logits: torch.Tensor, targets: torch.Tensor, lib: ctypes.CDLL, kernel_variant: str):
        if logits.device.type != "cpu" or targets.device.type != "cpu":
            raise RuntimeError("CCrossEntropyFn supports CPU tensors only")
        logits_np = logits.detach().contiguous().numpy().astype(np.float32, copy=False)
        targets_np = targets.detach().contiguous().numpy().astype(np.int32, copy=False)

        n, v = logits_np.shape
        dlogits_np = np.empty_like(logits_np, dtype=np.float32)
        loss_c = ctypes.c_float(0.0)

        variant = str(kernel_variant).lower().strip()
        if variant == "ptref":
            kernel_fn = getattr(lib, "softmax_cross_entropy_loss_ptref", None)
            if kernel_fn is None:
                raise RuntimeError(
                    "Requested --ck-loss-backend c_ptref but symbol "
                    "`softmax_cross_entropy_loss_ptref` is missing. "
                    "Rebuild shared library via `make build/libckernel_engine.so`."
                )
        else:
            kernel_fn = lib.softmax_cross_entropy_loss

        kernel_fn(
            _float_ptr(logits_np),
            _int32_ptr(targets_np),
            ctypes.c_int(n),
            ctypes.c_int(v),
            _float_ptr(dlogits_np),
            ctypes.byref(loss_c),
        )

        # Own the gradient buffer to avoid any NumPy allocator alias/lifetime surprises.
        ctx.dlogits = torch.from_numpy(dlogits_np.copy())
        return logits.new_tensor(loss_c.value)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        # dL/dlogits was already produced for mean CE; scale by upstream scalar.
        return ctx.dlogits * grad_out.to(dtype=ctx.dlogits.dtype), None, None, None


def c_rmsnorm(x: torch.Tensor, gamma: torch.Tensor, eps: float, lib: ctypes.CDLL) -> torch.Tensor:
    return CRMSNormFn.apply(x, gamma, eps, lib)


def c_swiglu(x: torch.Tensor, lib: ctypes.CDLL) -> torch.Tensor:
    return CSwiGLUFn.apply(x, lib)


def _is_c_loss_backend(name: str) -> bool:
    return str(name).lower().strip() in {"c", "c_ptref"}


def _c_loss_kernel_variant(name: str) -> str:
    loss_backend = str(name).lower().strip()
    if loss_backend == "c_ptref":
        return "ptref"
    return "default"


def c_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    lib: ctypes.CDLL,
    *,
    kernel_variant: str = "default",
) -> torch.Tensor:
    return CCrossEntropyFn.apply(logits, targets, lib, kernel_variant)


class TinyCKModel(nn.Module):
    def __init__(
        self,
        vocab: int,
        d_model: int,
        hidden: int,
        eps: float,
        lib: ctypes.CDLL,
        use_c_rmsnorm: bool = True,
        use_c_swiglu: bool = True,
    ):
        super().__init__()
        self.vocab = vocab
        self.d_model = d_model
        self.hidden = hidden
        self.eps = eps
        self.lib = lib
        self.use_c_rmsnorm = bool(use_c_rmsnorm)
        self.use_c_swiglu = bool(use_c_swiglu)

        self.embedding = nn.Embedding(vocab, d_model)
        self.rms_gamma = nn.Parameter(torch.ones(d_model, dtype=torch.float32))
        self.fc1 = nn.Linear(d_model, 2 * hidden)
        self.fc2 = nn.Linear(hidden, vocab)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # input_ids: [B, T] -> logits: [B*T, V]
        x = self.embedding(input_ids).reshape(-1, self.d_model)
        if self.use_c_rmsnorm:
            x = c_rmsnorm(x, self.rms_gamma, self.eps, self.lib)
        else:
            var = x.pow(2).mean(dim=-1, keepdim=True)
            rstd = (var + self.eps).rsqrt()
            x = x * rstd * self.rms_gamma
        x = self.fc1(x)
        if self.use_c_swiglu:
            x = c_swiglu(x, self.lib)
        else:
            gate, up = x[:, : self.hidden], x[:, self.hidden :]
            x = F.silu(gate) * up
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


def _build_batches_from_text(text: str, total_tokens: int, seq_len: int, vocab: int) -> list[Tuple[torch.Tensor, torch.Tensor]]:
    data = (text or "").encode("utf-8", errors="ignore")
    if len(data) < 2:
        raise ValueError("--train-text must encode to at least 2 bytes")
    ids = [int(b) % int(vocab) for b in data]
    needed = int(total_tokens) + 1
    repeats = (needed + len(ids) - 1) // len(ids)
    stream = np.array((ids * repeats)[:needed], dtype=np.int64)
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



def _manifest_entries_map(manifest: dict) -> Dict[str, dict]:
    entries = manifest.get("entries")
    if not isinstance(entries, list):
        raise ValueError("Invalid manifest: missing entries[]")
    return {str(e.get("name")): e for e in entries if isinstance(e, dict) and e.get("name")}


def _load_tensor_from_bump(entry: dict, bump_blob: bytes) -> np.ndarray:
    if str(entry.get("dtype", "")).lower() != "fp32":
        raise ValueError(f"Unsupported dtype for {entry.get('name')}: {entry.get('dtype')} (expected fp32)")
    off = int(entry.get("offset", 0))
    size = int(entry.get("size", 0))
    shape = tuple(int(x) for x in (entry.get("shape") or []))
    if off < 0 or size <= 0 or off + size > len(bump_blob):
        raise ValueError(f"Invalid tensor span for {entry.get('name')}: off={off} size={size}")
    arr = np.frombuffer(bump_blob, dtype=np.float32, count=size // 4, offset=off).copy()
    if shape:
        arr = arr.reshape(shape)
    return arr


def _load_tiny_state_from_bump(weights_bump: Path, weights_manifest: Path) -> tuple[Dict[str, torch.Tensor], dict]:
    manifest = json.loads(weights_manifest.read_text(encoding="utf-8"))
    entries = _manifest_entries_map(manifest)
    bump_blob = weights_bump.read_bytes()

    required = {
        "embedding.weight": "tiny.embedding.weight",
        "rms_gamma": "tiny.rms_gamma",
        "fc1.weight": "tiny.fc1.weight",
        "fc1.bias": "tiny.fc1.bias",
        "fc2.weight": "tiny.fc2.weight",
        "fc2.bias": "tiny.fc2.bias",
    }
    missing = [tensor for tensor in required.values() if tensor not in entries]
    if missing:
        raise ValueError(
            "weights.bump/manifest missing tiny parity tensors. "
            f"Missing: {', '.join(missing)}"
        )

    state: Dict[str, torch.Tensor] = {}
    for state_name, manifest_name in required.items():
        arr = _load_tensor_from_bump(entries[manifest_name], bump_blob)
        state[state_name] = torch.from_numpy(arr.astype(np.float32, copy=False))

    embed_shape = tuple(state["embedding.weight"].shape)
    fc1_shape = tuple(state["fc1.weight"].shape)
    fc2_shape = tuple(state["fc2.weight"].shape)
    dims = {
        "vocab": int(embed_shape[0]),
        "d_model": int(embed_shape[1]),
        "hidden": int(fc1_shape[0] // 2),
    }
    if int(fc1_shape[1]) != dims["d_model"]:
        raise ValueError(f"tiny.fc1.weight shape mismatch: expected [2H, D], got {fc1_shape}")
    if int(fc2_shape[0]) != dims["vocab"] or int(fc2_shape[1]) != dims["hidden"]:
        raise ValueError(
            "tiny.fc2.weight shape mismatch: "
            f"expected [{dims['vocab']}, {dims['hidden']}], got {fc2_shape}"
        )
    if tuple(state["fc1.bias"].shape) != (2 * dims["hidden"],):
        raise ValueError("tiny.fc1.bias shape mismatch")
    if tuple(state["fc2.bias"].shape) != (dims["vocab"],):
        raise ValueError("tiny.fc2.bias shape mismatch")
    if tuple(state["rms_gamma"].shape) != (dims["d_model"],):
        raise ValueError("tiny.rms_gamma shape mismatch")

    return state, dims

def _global_grad_norm(params: Iterable[torch.nn.Parameter]) -> float:
    total = 0.0
    for p in params:
        if p.grad is None:
            continue
        g = p.grad.detach().float()
        total += float(torch.sum(g * g).item())
    return math.sqrt(total) if total > 0.0 else 0.0


def _param_grad_norms(named_params: Iterable[Tuple[str, torch.nn.Parameter]]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for name, p in named_params:
        if p.grad is None:
            out[name] = 0.0
        else:
            out[name] = float(torch.norm(p.grad.detach().float()).item())
    return out


def _state_dict_diff_stats(
    ck_state: Dict[str, torch.Tensor],
    torch_state: Dict[str, torch.Tensor],
) -> Tuple[float, float, str]:
    max_diff = 0.0
    mean_acc = 0.0
    count = 0
    worst = "n/a"
    for name, p_ck in ck_state.items():
        p_t = torch_state.get(name)
        if p_t is None:
            continue
        d = (p_ck - p_t).abs()
        dmax = float(d.max().item()) if d.numel() else 0.0
        dmean = float(d.mean().item()) if d.numel() else 0.0
        if dmax > max_diff:
            max_diff = dmax
            worst = name
        mean_acc += dmean
        count += 1
    mean_diff = mean_acc / count if count > 0 else 0.0
    return max_diff, mean_diff, worst


def _topk_state_dict_diffs(
    ck_state: Dict[str, torch.Tensor],
    torch_state: Dict[str, torch.Tensor],
    k: int,
) -> list[dict]:
    rows: list[dict] = []
    topk = max(1, int(k))
    for name, p_ck in ck_state.items():
        p_t = torch_state.get(name)
        if p_t is None:
            continue
        d = (p_ck - p_t).abs()
        dmax = float(d.max().item()) if d.numel() else 0.0
        dmean = float(d.mean().item()) if d.numel() else 0.0
        dl2 = float(torch.norm(d.float()).item()) if d.numel() else 0.0
        rows.append(
            {
                "name": name,
                "max_abs_diff": dmax,
                "mean_abs_diff": dmean,
                "l2_diff": dl2,
            }
        )
    rows.sort(key=lambda r: float(r["max_abs_diff"]), reverse=True)
    return rows[:topk]


def _named_tensor_diff_stats(
    ck_named: Iterable[Tuple[str, torch.Tensor]],
    torch_named: Iterable[Tuple[str, torch.Tensor]],
) -> Tuple[float, float, str]:
    max_diff = 0.0
    mean_acc = 0.0
    count = 0
    worst = "n/a"
    for (name_ck, t_ck), (name_t, t_t) in zip(ck_named, torch_named):
        if name_ck != name_t:
            raise RuntimeError(f"Tensor name mismatch: {name_ck} vs {name_t}")
        d = (t_ck - t_t).abs()
        dmax = float(d.max().item()) if d.numel() else 0.0
        dmean = float(d.mean().item()) if d.numel() else 0.0
        if dmax > max_diff:
            max_diff = dmax
            worst = name_ck
        mean_acc += dmean
        count += 1
    mean_diff = mean_acc / count if count > 0 else 0.0
    return max_diff, mean_diff, worst


def _optimizer_state_diff_stats(
    model_ck: nn.Module,
    model_torch: nn.Module,
    opt_ck: torch.optim.Optimizer,
    opt_torch: torch.optim.Optimizer,
) -> dict:
    name_to_p_ck = {name: p for name, p in model_ck.named_parameters()}
    name_to_p_t = {name: p for name, p in model_torch.named_parameters()}
    max_exp_avg_diff = 0.0
    max_exp_avg_sq_diff = 0.0
    worst_exp_avg = "n/a"
    worst_exp_avg_sq = "n/a"
    for name, p_ck in name_to_p_ck.items():
        p_t = name_to_p_t.get(name)
        if p_t is None:
            continue
        state_ck = opt_ck.state.get(p_ck, {})
        state_t = opt_torch.state.get(p_t, {})
        m_ck = state_ck.get("exp_avg")
        m_t = state_t.get("exp_avg")
        if m_ck is not None and m_t is not None:
            d = (m_ck.detach() - m_t.detach()).abs()
            dmax = float(d.max().item()) if d.numel() else 0.0
            if dmax > max_exp_avg_diff:
                max_exp_avg_diff = dmax
                worst_exp_avg = name
        v_ck = state_ck.get("exp_avg_sq")
        v_t = state_t.get("exp_avg_sq")
        if v_ck is not None and v_t is not None:
            d = (v_ck.detach() - v_t.detach()).abs()
            dmax = float(d.max().item()) if d.numel() else 0.0
            if dmax > max_exp_avg_sq_diff:
                max_exp_avg_sq_diff = dmax
                worst_exp_avg_sq = name
    return {
        "max_exp_avg_diff": max_exp_avg_diff,
        "max_exp_avg_sq_diff": max_exp_avg_sq_diff,
        "worst_exp_avg_param": worst_exp_avg,
        "worst_exp_avg_sq_param": worst_exp_avg_sq,
    }


_STAGE_ORDER = ["embedding", "rmsnorm", "fc1", "swiglu", "logits", "loss"]
_BWD_STAGE_ORDER = ["logits", "swiglu", "fc1", "rmsnorm", "embedding"]


def _forward_stages_ck(
    model_ck: TinyCKModel,
    x: torch.Tensor,
    targets: torch.Tensor,
    eps: float,
    lib: ctypes.CDLL,
    ck_loss_backend: str,
) -> dict[str, torch.Tensor]:
    stages: dict[str, torch.Tensor] = {}
    emb = model_ck.embedding(x).reshape(-1, model_ck.d_model)
    stages["embedding"] = emb

    if model_ck.use_c_rmsnorm:
        rms = c_rmsnorm(emb, model_ck.rms_gamma, eps, lib)
    else:
        var = emb.pow(2).mean(dim=-1, keepdim=True)
        rstd = (var + eps).rsqrt()
        rms = emb * rstd * model_ck.rms_gamma
    stages["rmsnorm"] = rms

    fc1 = model_ck.fc1(rms)
    stages["fc1"] = fc1

    if model_ck.use_c_swiglu:
        sw = c_swiglu(fc1, lib)
    else:
        gate, up = fc1[:, : model_ck.hidden], fc1[:, model_ck.hidden :]
        sw = F.silu(gate) * up
    stages["swiglu"] = sw

    logits = model_ck.fc2(sw)
    stages["logits"] = logits

    if _is_c_loss_backend(ck_loss_backend):
        loss = c_cross_entropy(
            logits,
            targets,
            lib,
            kernel_variant=_c_loss_kernel_variant(ck_loss_backend),
        )
    else:
        loss = F.cross_entropy(logits, targets, reduction="mean")
    stages["loss"] = loss
    return stages


def _forward_stages_torch(
    model_torch: TinyTorchModel,
    x: torch.Tensor,
    targets: torch.Tensor,
    eps: float,
) -> dict[str, torch.Tensor]:
    stages: dict[str, torch.Tensor] = {}
    emb = model_torch.embedding(x).reshape(-1, model_torch.d_model)
    stages["embedding"] = emb

    var = emb.pow(2).mean(dim=-1, keepdim=True)
    rstd = (var + eps).rsqrt()
    rms = emb * rstd * model_torch.rms_gamma
    stages["rmsnorm"] = rms

    fc1 = model_torch.fc1(rms)
    stages["fc1"] = fc1

    gate, up = fc1[:, : model_torch.hidden], fc1[:, model_torch.hidden :]
    sw = F.silu(gate) * up
    stages["swiglu"] = sw

    logits = model_torch.fc2(sw)
    stages["logits"] = logits
    stages["loss"] = F.cross_entropy(logits, targets, reduction="mean")
    return stages


def _scalar_or_max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.ndim == 0 and b.ndim == 0:
        return abs(float(a.item()) - float(b.item()))
    d = (a.detach() - b.detach()).abs()
    return float(d.max().item()) if d.numel() else 0.0


def _localize_step_divergence(
    lib: ctypes.CDLL,
    window_samples: list[tuple[torch.Tensor, torch.Tensor]],
    vocab: int,
    d_model: int,
    hidden: int,
    eps: float,
    optimizer: str,
    lr: float,
    grad_accum: int,
    ck_rmsnorm_backend: str,
    ck_swiglu_backend: str,
    ck_loss_backend: str,
    pre_ck_model_state: Dict[str, torch.Tensor],
    pre_ck_opt_state: dict,
    pre_torch_model_state: Dict[str, torch.Tensor],
    pre_torch_opt_state: dict,
    source: str,
    tol: float,
) -> dict:
    report: dict = {
        "same_state_source": str(source),
        "tol": float(tol),
        "micro_steps": int(len(window_samples)),
        "stage_order": list(_STAGE_ORDER),
    }

    if not window_samples:
        report["error"] = "no_window_samples"
        return report

    ck_probe = TinyCKModel(
        vocab=vocab,
        d_model=d_model,
        hidden=hidden,
        eps=eps,
        lib=lib,
        use_c_rmsnorm=(str(ck_rmsnorm_backend).lower() == "c"),
        use_c_swiglu=(str(ck_swiglu_backend).lower() == "c"),
    )
    torch_probe = TinyTorchModel(vocab=vocab, d_model=d_model, hidden=hidden, eps=eps)

    opt_ck_probe = _make_optimizer(optimizer, ck_probe.parameters(), lr=lr)
    opt_torch_probe = _make_optimizer(optimizer, torch_probe.parameters(), lr=lr)

    try:
        if str(source).lower() == "torch":
            base_model_state = copy.deepcopy(pre_torch_model_state)
            base_opt_state = copy.deepcopy(pre_torch_opt_state)
        else:
            base_model_state = copy.deepcopy(pre_ck_model_state)
            base_opt_state = copy.deepcopy(pre_ck_opt_state)

        ck_probe.load_state_dict(base_model_state, strict=True)
        torch_probe.load_state_dict(base_model_state, strict=True)

        opt_ck_probe.load_state_dict(copy.deepcopy(base_opt_state))
        opt_torch_probe.load_state_dict(copy.deepcopy(base_opt_state))
    except Exception as e:
        report["error"] = f"state_restore_failed: {e}"
        return report

    ck_probe.train()
    torch_probe.train()
    opt_ck_probe.zero_grad(set_to_none=True)
    opt_torch_probe.zero_grad(set_to_none=True)

    stage_max_global = {name: 0.0 for name in _STAGE_ORDER}
    stage_grad_max_global = {name: 0.0 for name in _BWD_STAGE_ORDER}
    per_micro: list[dict] = []

    for idx, (x_i, targets_i) in enumerate(window_samples, start=1):
        stages_ck = _forward_stages_ck(ck_probe, x_i, targets_i, eps, lib, ck_loss_backend)
        stages_torch = _forward_stages_torch(torch_probe, x_i, targets_i, eps)

        stage_diffs: dict[str, float] = {}
        first_stage_over_tol = "none"
        for stage in _STAGE_ORDER:
            diff = _scalar_or_max_abs_diff(stages_ck[stage], stages_torch[stage])
            stage_diffs[stage] = float(diff)
            if diff > stage_max_global[stage]:
                stage_max_global[stage] = float(diff)
            if first_stage_over_tol == "none" and diff > tol:
                first_stage_over_tol = stage

        for stage in _BWD_STAGE_ORDER:
            t_ck = stages_ck.get(stage)
            t_torch = stages_torch.get(stage)
            if isinstance(t_ck, torch.Tensor) and t_ck.requires_grad:
                t_ck.retain_grad()
            if isinstance(t_torch, torch.Tensor) and t_torch.requires_grad:
                t_torch.retain_grad()

        (stages_ck["loss"] / float(grad_accum)).backward()
        (stages_torch["loss"] / float(grad_accum)).backward()

        stage_grad_diffs: dict[str, float] = {}
        first_grad_stage_over_tol = "none"
        for stage in _BWD_STAGE_ORDER:
            t_ck = stages_ck.get(stage)
            t_torch = stages_torch.get(stage)
            g_ck = t_ck.grad if isinstance(t_ck, torch.Tensor) else None
            g_torch = t_torch.grad if isinstance(t_torch, torch.Tensor) else None
            if g_ck is None and g_torch is None:
                diff = 0.0
            elif g_ck is None or g_torch is None:
                diff = float("inf")
            else:
                diff = _scalar_or_max_abs_diff(g_ck, g_torch)
            stage_grad_diffs[stage] = float(diff)
            if diff > stage_grad_max_global[stage]:
                stage_grad_max_global[stage] = float(diff)
            if first_grad_stage_over_tol == "none" and diff > tol:
                first_grad_stage_over_tol = stage

        per_micro.append(
            {
                "micro_step": int(idx),
                "first_stage_over_tol": first_stage_over_tol,
                "stage_max_diffs": stage_diffs,
                "first_grad_stage_over_tol": first_grad_stage_over_tol,
                "stage_grad_max_diffs": stage_grad_diffs,
            }
        )

    grad_max, grad_mean, worst_grad = _named_tensor_diff_stats(
        ((name, p.grad.detach()) for name, p in ck_probe.named_parameters() if p.grad is not None),
        ((name, p.grad.detach()) for name, p in torch_probe.named_parameters() if p.grad is not None),
    )

    pre_param_max, pre_param_mean, pre_worst = _state_dict_diff_stats(
        ck_probe.state_dict(), torch_probe.state_dict()
    )
    pre_opt_diag = _optimizer_state_diff_stats(ck_probe, torch_probe, opt_ck_probe, opt_torch_probe)

    opt_ck_probe.step()
    opt_torch_probe.step()
    opt_ck_probe.zero_grad(set_to_none=True)
    opt_torch_probe.zero_grad(set_to_none=True)

    post_param_max, post_param_mean, post_worst = _state_dict_diff_stats(
        ck_probe.state_dict(), torch_probe.state_dict()
    )
    post_opt_diag = _optimizer_state_diff_stats(ck_probe, torch_probe, opt_ck_probe, opt_torch_probe)

    first_stage_global = "none"
    for stage in _STAGE_ORDER:
        if float(stage_max_global.get(stage, 0.0)) > tol:
            first_stage_global = stage
            break

    first_grad_stage_global = "none"
    for stage in _BWD_STAGE_ORDER:
        if float(stage_grad_max_global.get(stage, 0.0)) > tol:
            first_grad_stage_global = stage
            break

    report.update(
        {
            "first_stage_over_tol": first_stage_global,
            "first_grad_stage_over_tol": first_grad_stage_global,
            "stage_order": list(_STAGE_ORDER),
            "stage_grad_order": list(_BWD_STAGE_ORDER),
            "stage_max_global": stage_max_global,
            "stage_grad_max_global": stage_grad_max_global,
            "grad_window": {
                "max_grad_diff": float(grad_max),
                "mean_grad_diff": float(grad_mean),
                "worst_grad_param": worst_grad,
            },
            "pre_step_same_state": {
                "max_param_diff": float(pre_param_max),
                "mean_param_diff": float(pre_param_mean),
                "worst_param": pre_worst,
                **pre_opt_diag,
            },
            "post_step_same_state": {
                "max_param_diff": float(post_param_max),
                "mean_param_diff": float(post_param_mean),
                "worst_param": post_worst,
                **post_opt_diag,
            },
            "per_micro": per_micro,
        }
    )
    return report


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
    loss_curve: list[dict]
    parity_steps: list[dict]
    grad_norm_series: dict
    step_profile: dict
    drift_diagnostics: dict
    epoch_snapshots: list[dict]
    safety: dict


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
    train_text: str | None = None,
    init_state: Dict[str, torch.Tensor] | None = None,
    max_steps: int | None = None,
    diag_every: int = 1,
    ck_rmsnorm_backend: str = "c",
    ck_swiglu_backend: str = "c",
    ck_loss_backend: str = "c",
    drift_localize_step: int = 0,
    drift_localize_tol: float = 1e-6,
    drift_localize_source: str = "ck",
    epoch_snapshot_every: int = 1,
    epoch_snapshot_topk: int = 8,
    max_grad_norm: float = 0.0,
    safety: dict | None = None,
) -> RunStats:
    _seed_all(seed)

    model_ck = TinyCKModel(
        vocab=vocab,
        d_model=d_model,
        hidden=hidden,
        eps=eps,
        lib=lib,
        use_c_rmsnorm=(str(ck_rmsnorm_backend).lower() == "c"),
        use_c_swiglu=(str(ck_swiglu_backend).lower() == "c"),
    )
    model_torch = TinyTorchModel(vocab=vocab, d_model=d_model, hidden=hidden, eps=eps)
    if init_state is not None:
        model_ck.load_state_dict(init_state, strict=True)
        model_torch.load_state_dict(init_state, strict=True)
    else:
        model_torch.load_state_dict(model_ck.state_dict(), strict=True)

    params_ck = list(model_ck.parameters())
    params_torch = list(model_torch.parameters())
    opt_ck = _make_optimizer(optimizer, params_ck, lr=lr)
    opt_torch = _make_optimizer(optimizer, params_torch, lr=lr)

    if train_text:
        batches = _build_batches_from_text(train_text, total_tokens=total_tokens, seq_len=seq_len, vocab=vocab)
    else:
        batches = _build_batches(total_tokens=total_tokens, seq_len=seq_len, vocab=vocab, seed=seed + 1)
    if not batches:
        raise RuntimeError("No batches generated; increase --total-tokens or reduce --seq-len")

    loss_diffs: list[float] = []
    last_ck = math.nan
    last_t = math.nan
    micro_count = 0
    step_count = 0
    processed_tokens = 0
    total_ck_ms = 0.0
    total_torch_ms = 0.0

    # Aggregators over one optimizer window (grad_accum micro-steps).
    win_micro = 0
    win_loss_ck_sum = 0.0
    win_loss_t_sum = 0.0
    win_loss_diff_max = 0.0
    win_logit_diff_max = 0.0
    win_tokens = 0
    win_ck_forward_ms = 0.0
    win_ck_backward_ms = 0.0
    win_ck_opt_ms = 0.0
    win_torch_forward_ms = 0.0
    win_torch_backward_ms = 0.0
    win_torch_opt_ms = 0.0

    loss_curve: list[dict] = []
    parity_steps: list[dict] = []
    grad_steps: list[int] = []
    grad_global: list[float] = []
    grad_params: Dict[str, list[float]] = {}
    diag_steps: list[dict] = []
    epoch_snapshots: list[dict] = []
    window_samples: list[tuple[torch.Tensor, torch.Tensor]] = []
    localized_step_report: dict | None = None

    def _flush_optimizer_step() -> None:
        nonlocal step_count
        nonlocal win_micro, win_loss_ck_sum, win_loss_t_sum, win_loss_diff_max, win_logit_diff_max, win_tokens
        nonlocal win_ck_forward_ms, win_ck_backward_ms, win_ck_opt_ms
        nonlocal win_torch_forward_ms, win_torch_backward_ms, win_torch_opt_ms
        nonlocal total_ck_ms, total_torch_ms
        nonlocal window_samples, localized_step_report
        if win_micro == 0:
            return

        target_step = int(step_count + 1)
        pre_param_max, pre_param_mean, pre_worst_param = _state_dict_diff_stats(
            model_ck.state_dict(), model_torch.state_dict()
        )
        pre_opt_state_diag = _optimizer_state_diff_stats(model_ck, model_torch, opt_ck, opt_torch)
        pre_opt_state_diag_prefixed = {
            f"pre_{k}": (float(v) if isinstance(v, (int, float)) else v)
            for k, v in pre_opt_state_diag.items()
        }

        if (
            int(drift_localize_step or 0) > 0
            and target_step == int(drift_localize_step)
            and localized_step_report is None
        ):
            pre_ck_model_state = copy.deepcopy(model_ck.state_dict())
            pre_torch_model_state = copy.deepcopy(model_torch.state_dict())
            pre_ck_opt_state = copy.deepcopy(opt_ck.state_dict())
            pre_torch_opt_state = copy.deepcopy(opt_torch.state_dict())
            localized_step_report = _localize_step_divergence(
                lib=lib,
                window_samples=window_samples,
                vocab=vocab,
                d_model=d_model,
                hidden=hidden,
                eps=eps,
                optimizer=optimizer,
                lr=lr,
                grad_accum=grad_accum,
                ck_rmsnorm_backend=ck_rmsnorm_backend,
                ck_swiglu_backend=ck_swiglu_backend,
                ck_loss_backend=ck_loss_backend,
                pre_ck_model_state=pre_ck_model_state,
                pre_ck_opt_state=pre_ck_opt_state,
                pre_torch_model_state=pre_torch_model_state,
                pre_torch_opt_state=pre_torch_opt_state,
                source=drift_localize_source,
                tol=float(drift_localize_tol),
            )
            localized_step_report["target_step"] = target_step
            localized_step_report["trajectory_pre_step"] = {
                "max_param_diff": float(pre_param_max),
                "mean_param_diff": float(pre_param_mean),
                "worst_param": pre_worst_param,
                **pre_opt_state_diag,
            }

        grad_max, grad_mean, worst_grad_param = _named_tensor_diff_stats(
            ((name, p.grad.detach()) for name, p in model_ck.named_parameters() if p.grad is not None),
            ((name, p.grad.detach()) for name, p in model_torch.named_parameters() if p.grad is not None),
        )
        grad_norm_pre_clip = _global_grad_norm(params_ck)
        torch_grad_norm_pre_clip = _global_grad_norm(params_torch)
        grad_norm = grad_norm_pre_clip
        per_param = _param_grad_norms(model_ck.named_parameters())
        clip_applied = False
        clip_total_ck = grad_norm_pre_clip
        clip_total_torch = torch_grad_norm_pre_clip
        clip_scale_ck = 1.0
        clip_scale_torch = 1.0
        if max_grad_norm > 0.0:
            clip_applied = True
            clip_total_ck = float(torch.nn.utils.clip_grad_norm_(params_ck, max_grad_norm).item())
            clip_total_torch = float(torch.nn.utils.clip_grad_norm_(params_torch, max_grad_norm).item())
            if clip_total_ck > float(max_grad_norm) and clip_total_ck > 0.0:
                clip_scale_ck = float(max_grad_norm / clip_total_ck)
            if clip_total_torch > float(max_grad_norm) and clip_total_torch > 0.0:
                clip_scale_torch = float(max_grad_norm / clip_total_torch)
            grad_norm = _global_grad_norm(params_ck)

        t_opt_ck_0 = time.perf_counter()
        opt_ck.step()
        opt_ck.zero_grad(set_to_none=True)
        t_opt_ck_1 = time.perf_counter()

        t_opt_t_0 = time.perf_counter()
        opt_torch.step()
        opt_torch.zero_grad(set_to_none=True)
        t_opt_t_1 = time.perf_counter()

        win_ck_opt_ms += (t_opt_ck_1 - t_opt_ck_0) * 1000.0
        win_torch_opt_ms += (t_opt_t_1 - t_opt_t_0) * 1000.0
        step_count += 1

        param_max, param_mean, worst_param = _state_dict_diff_stats(
            model_ck.state_dict(), model_torch.state_dict()
        )
        opt_state_diag = _optimizer_state_diff_stats(model_ck, model_torch, opt_ck, opt_torch)
        step_loss_ck = win_loss_ck_sum / float(win_micro)
        step_loss_t = win_loss_t_sum / float(win_micro)
        lr_now = float(opt_ck.param_groups[0].get("lr", lr))

        step_ck_ms = win_ck_forward_ms + win_ck_backward_ms + win_ck_opt_ms
        step_torch_ms = win_torch_forward_ms + win_torch_backward_ms + win_torch_opt_ms
        total_ck_ms += step_ck_ms
        total_torch_ms += step_torch_ms

        loss_curve.append(
            {
                "step": step_count,
                "micro_steps": win_micro,
                "tokens": win_tokens,
                "loss_ck": step_loss_ck,
                "loss_pt": step_loss_t,
                "lr": lr_now,
                "grad_norm": grad_norm,
                "forward_ms": win_ck_forward_ms,
                "backward_ms": win_ck_backward_ms,
                "optimizer_ms": win_ck_opt_ms,
                "step_ms": step_ck_ms,
                "torch_forward_ms": win_torch_forward_ms,
                "torch_backward_ms": win_torch_backward_ms,
                "torch_optimizer_ms": win_torch_opt_ms,
                "torch_step_ms": step_torch_ms,
                "grad_norm_pre_clip": grad_norm_pre_clip,
                "torch_grad_norm_pre_clip": torch_grad_norm_pre_clip,
                "grad_clip_applied": bool(clip_applied),
                "max_grad_norm": float(max_grad_norm),
            }
        )
        parity_steps.append(
            {
                "step": step_count,
                "loss_diff": win_loss_diff_max,
                "max_logit_diff": win_logit_diff_max,
                "max_grad_diff": grad_max,
                "mean_grad_diff": grad_mean,
                "worst_grad_param": worst_grad_param,
                "max_param_diff": param_max,
                "worst_param": worst_param,
                "mean_param_diff": param_mean,
                "pre_max_param_diff": pre_param_max,
                "pre_mean_param_diff": pre_param_mean,
                "pre_worst_param": pre_worst_param,
                **pre_opt_state_diag_prefixed,
                **opt_state_diag,
                "grad_norm_pre_clip": float(grad_norm_pre_clip),
                "torch_grad_norm_pre_clip": float(torch_grad_norm_pre_clip),
                "grad_clip_applied": bool(clip_applied),
                "max_grad_norm": float(max_grad_norm),
                "clip_total_norm_ck": float(clip_total_ck),
                "clip_total_norm_torch": float(clip_total_torch),
                "clip_scale_ck": float(clip_scale_ck),
                "clip_scale_torch": float(clip_scale_torch),
            }
        )
        if diag_every > 0 and (step_count % diag_every == 0):
            diag_steps.append(
                {
                    "step": step_count,
                    "loss_diff": win_loss_diff_max,
                    "max_logit_diff": win_logit_diff_max,
                    "max_grad_diff": grad_max,
                    "worst_grad_param": worst_grad_param,
                    "max_param_diff": param_max,
                    "worst_param": worst_param,
                    "pre_max_param_diff": pre_param_max,
                    "pre_worst_param": pre_worst_param,
                    **pre_opt_state_diag_prefixed,
                    **opt_state_diag,
                    "grad_norm_pre_clip": float(grad_norm_pre_clip),
                    "torch_grad_norm_pre_clip": float(torch_grad_norm_pre_clip),
                    "grad_clip_applied": bool(clip_applied),
                    "max_grad_norm": float(max_grad_norm),
                    "clip_total_norm_ck": float(clip_total_ck),
                    "clip_total_norm_torch": float(clip_total_torch),
                    "clip_scale_ck": float(clip_scale_ck),
                    "clip_scale_torch": float(clip_scale_torch),
                }
            )
        grad_steps.append(step_count)
        grad_global.append(grad_norm)
        for name, value in per_param.items():
            grad_params.setdefault(name, []).append(float(value))

        win_micro = 0
        win_loss_ck_sum = 0.0
        win_loss_t_sum = 0.0
        win_loss_diff_max = 0.0
        win_logit_diff_max = 0.0
        win_tokens = 0
        win_ck_forward_ms = 0.0
        win_ck_backward_ms = 0.0
        win_ck_opt_ms = 0.0
        win_torch_forward_ms = 0.0
        win_torch_backward_ms = 0.0
        win_torch_opt_ms = 0.0
        window_samples = []

    def _record_epoch_snapshot(epoch_idx: int, step_begin: int) -> None:
        ck_state = model_ck.state_dict()
        torch_state = model_torch.state_dict()
        param_max, param_mean, worst_param = _state_dict_diff_stats(ck_state, torch_state)
        opt_diag = _optimizer_state_diff_stats(model_ck, model_torch, opt_ck, opt_torch)
        step_slice = parity_steps[int(step_begin) : int(step_count)]
        loss_slice = loss_curve[int(step_begin) : int(step_count)]
        epoch_snapshots.append(
            {
                "epoch": int(epoch_idx),
                "optimizer_step_start": int(step_begin + 1) if step_count > step_begin else int(step_begin),
                "optimizer_step_end": int(step_count),
                "optimizer_steps_in_epoch": int(max(0, step_count - step_begin)),
                "loss_ck_last": float(last_ck) if not math.isnan(last_ck) else math.nan,
                "loss_torch_last": float(last_t) if not math.isnan(last_t) else math.nan,
                "loss_abs_diff_last": (
                    abs(float(last_ck) - float(last_t))
                    if not (math.isnan(last_ck) or math.isnan(last_t))
                    else math.nan
                ),
                "max_loss_diff_in_epoch": max((float(s["loss_diff"]) for s in step_slice), default=0.0),
                "max_logit_diff_in_epoch": max((float(s.get("max_logit_diff", 0.0)) for s in step_slice), default=0.0),
                "max_grad_diff_in_epoch": max((float(s.get("max_grad_diff", 0.0)) for s in step_slice), default=0.0),
                "mean_ck_step_ms_in_epoch": (
                    sum(float(s.get("step_ms", 0.0)) for s in loss_slice) / len(loss_slice)
                    if loss_slice
                    else 0.0
                ),
                "mean_torch_step_ms_in_epoch": (
                    sum(float(s.get("torch_step_ms", 0.0)) for s in loss_slice) / len(loss_slice)
                    if loss_slice
                    else 0.0
                ),
                "max_param_diff": float(param_max),
                "mean_param_diff": float(param_mean),
                "worst_param": worst_param,
                **opt_diag,
                "top_param_diffs": _topk_state_dict_diffs(ck_state, torch_state, k=epoch_snapshot_topk),
            }
        )

    opt_ck.zero_grad(set_to_none=True)
    opt_torch.zero_grad(set_to_none=True)

    step_limit = int(max_steps or 0)

    for _epoch in range(epochs):
        epoch_step_start = int(step_count)
        for x, y in batches:
            targets = y.reshape(-1)
            window_samples.append((x.detach().clone(), targets.detach().clone()))

            t_ck_fwd_0 = time.perf_counter()
            logits_ck = model_ck(x)
            if _is_c_loss_backend(ck_loss_backend):
                loss_ck = c_cross_entropy(
                    logits_ck,
                    targets,
                    lib,
                    kernel_variant=_c_loss_kernel_variant(ck_loss_backend),
                )
            else:
                loss_ck = F.cross_entropy(logits_ck, targets, reduction="mean")
            t_ck_fwd_1 = time.perf_counter()

            t_t_fwd_0 = time.perf_counter()
            logits_t = model_torch(x)
            loss_t = F.cross_entropy(logits_t, targets, reduction="mean")
            logit_diff = float((logits_ck.detach() - logits_t.detach()).abs().max().item())
            t_t_fwd_1 = time.perf_counter()

            last_ck = float(loss_ck.item())
            last_t = float(loss_t.item())
            loss_diff = abs(last_ck - last_t)
            loss_diffs.append(loss_diff)

            t_ck_bwd_0 = time.perf_counter()
            (loss_ck / grad_accum).backward()
            t_ck_bwd_1 = time.perf_counter()
            t_t_bwd_0 = time.perf_counter()
            (loss_t / grad_accum).backward()
            t_t_bwd_1 = time.perf_counter()

            tokens_here = int(x.numel())
            processed_tokens += tokens_here
            micro_count += 1
            win_micro += 1
            win_tokens += tokens_here
            win_loss_ck_sum += last_ck
            win_loss_t_sum += last_t
            win_loss_diff_max = max(win_loss_diff_max, loss_diff)
            win_logit_diff_max = max(win_logit_diff_max, logit_diff)
            win_ck_forward_ms += (t_ck_fwd_1 - t_ck_fwd_0) * 1000.0
            win_torch_forward_ms += (t_t_fwd_1 - t_t_fwd_0) * 1000.0
            win_ck_backward_ms += (t_ck_bwd_1 - t_ck_bwd_0) * 1000.0
            win_torch_backward_ms += (t_t_bwd_1 - t_t_bwd_0) * 1000.0

            if micro_count % grad_accum == 0:
                _flush_optimizer_step()
                if step_limit > 0 and step_count >= step_limit:
                    break
        if epoch_snapshot_every > 0 and ((_epoch + 1) % epoch_snapshot_every == 0):
            _record_epoch_snapshot(epoch_idx=_epoch + 1, step_begin=epoch_step_start)
        if step_limit > 0 and step_count >= step_limit:
            break

    if micro_count % grad_accum != 0 and (step_limit <= 0 or step_count < step_limit):
        _flush_optimizer_step()
    if epoch_snapshot_every > 0 and len(epoch_snapshots) == 0 and step_count > 0:
        _record_epoch_snapshot(epoch_idx=epochs, step_begin=0)

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
    train_tok_s = (processed_tokens / (total_ck_ms / 1000.0)) if total_ck_ms > 0 else 0.0
    avg_ck_step_ms = (total_ck_ms / step_count) if step_count > 0 else 0.0
    avg_torch_step_ms = (total_torch_ms / step_count) if step_count > 0 else 0.0
    first_loss_fail_step = next((int(s["step"]) for s in parity_steps if float(s["loss_diff"]) > float(loss_tol)), None)
    first_param_fail_step = next((int(s["step"]) for s in parity_steps if float(s["max_param_diff"]) > float(param_tol)), None)
    max_logit_abs_diff = max((float(s.get("max_logit_diff", 0.0)) for s in parity_steps), default=0.0)
    max_grad_abs_diff = max((float(s.get("max_grad_diff", 0.0)) for s in parity_steps), default=0.0)

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
        loss_curve=loss_curve,
        parity_steps=parity_steps,
        grad_norm_series={
            "steps": grad_steps,
            "global": grad_global,
            "params": grad_params,
        },
        step_profile={
            "steps": step_count,
            "micro_steps": micro_count,
            "tokens_per_update": seq_len * grad_accum,
            "processed_tokens": processed_tokens,
            "ck_total_ms": total_ck_ms,
            "torch_total_ms": total_torch_ms,
            "ck_avg_step_ms": avg_ck_step_ms,
            "torch_avg_step_ms": avg_torch_step_ms,
            "train_tok_s": train_tok_s,
            # Keep this alias for existing dashboard card compatibility.
            "decode_tok_s": train_tok_s,
        },
        drift_diagnostics={
            "first_loss_fail_step": first_loss_fail_step,
            "first_param_fail_step": first_param_fail_step,
            "max_logit_abs_diff": max_logit_abs_diff,
            "max_grad_abs_diff": max_grad_abs_diff,
            "steps": diag_steps,
            "localize_step_report": localized_step_report,
        },
        epoch_snapshots=epoch_snapshots,
        safety=dict(safety or {}),
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
    parser.add_argument("--param-tol", type=float, default=3e-5)
    parser.add_argument("--train-text", type=str, default=None,
                        help="Optional UTF-8 text to build deterministic training tokens (repeated to fill total-tokens)")
    parser.add_argument("--weights-bump", type=Path, default=None,
                        help="Optional weights.bump path. If set with --weights-manifest, tiny parity state is loaded from bump")
    parser.add_argument("--weights-manifest", type=Path, default=None,
                        help="Optional weights_manifest.json path paired with --weights-bump")
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=0,
        help="Optional cap on optimizer steps (0 = run full epochs).",
    )
    parser.add_argument(
        "--diag-every",
        type=int,
        default=1,
        help="Emit drift diagnostics every N optimizer steps (0 disables detailed step list).",
    )
    parser.add_argument("--ck-rmsnorm-backend", choices=["c", "torch"], default="c",
                        help="CK model RMSNorm path for parity harness (default: c)")
    parser.add_argument("--ck-swiglu-backend", choices=["c", "torch"], default="c",
                        help="CK model SwiGLU path for parity harness (default: c)")
    parser.add_argument("--ck-loss-backend", choices=["c", "c_ptref", "torch"], default="c",
                        help="CK model loss path for parity harness (default: c)")
    parser.add_argument("--drift-localize-step", type=int, default=0,
                        help="If >0, capture paired/same-state localization report at this optimizer step")
    parser.add_argument("--drift-localize-tol", type=float, default=1e-6,
                        help="Stage diff threshold used by same-state localizer")
    parser.add_argument("--drift-localize-source", choices=["ck", "torch"], default="ck",
                        help="Same-state replay base for localization (ck or torch pre-step state)")
    parser.add_argument(
        "--epoch-snapshot-every",
        type=int,
        default=1,
        help="Emit epoch-level CK-vs-Torch snapshots every N epochs (0 disables).",
    )
    parser.add_argument(
        "--epoch-snapshot-topk",
        type=int,
        default=8,
        help="Top-K parameter diff rows to keep in each epoch snapshot.",
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=0.0,
        help="Global grad-norm clip applied to both CK and Torch branches before optimizer step (0 disables).",
    )
    parser.add_argument(
        "--enforce-production-safety",
        action="store_true",
        help="Fail fast on known-unsafe long-horizon AdamW settings.",
    )
    parser.add_argument(
        "--allow-unsafe-adamw-lr",
        action="store_true",
        help="Allow high AdamW LR without clipping even when production safety is enforced.",
    )
    parser.add_argument(
        "--unsafe-adamw-lr-threshold",
        type=float,
        default=1e-3,
        help="LR threshold used by production safety checks for all-C AdamW path.",
    )
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
    if args.max_steps < 0:
        print("ERROR: --max-steps must be >= 0", file=sys.stderr)
        return 2
    if args.diag_every < 0:
        print("ERROR: --diag-every must be >= 0", file=sys.stderr)
        return 2
    if args.drift_localize_step < 0:
        print("ERROR: --drift-localize-step must be >= 0", file=sys.stderr)
        return 2
    if args.drift_localize_tol < 0:
        print("ERROR: --drift-localize-tol must be >= 0", file=sys.stderr)
        return 2
    if args.epoch_snapshot_every < 0:
        print("ERROR: --epoch-snapshot-every must be >= 0", file=sys.stderr)
        return 2
    if args.epoch_snapshot_topk < 1:
        print("ERROR: --epoch-snapshot-topk must be >= 1", file=sys.stderr)
        return 2
    if args.max_grad_norm < 0:
        print("ERROR: --max-grad-norm must be >= 0", file=sys.stderr)
        return 2
    if args.unsafe_adamw_lr_threshold <= 0:
        print("ERROR: --unsafe-adamw-lr-threshold must be > 0", file=sys.stderr)
        return 2

    risky_all_c_adamw = (
        str(args.optimizer).lower() == "adamw"
        and float(args.lr) >= float(args.unsafe_adamw_lr_threshold)
        and str(args.ck_rmsnorm_backend).lower() == "c"
        and str(args.ck_swiglu_backend).lower() == "c"
        and _is_c_loss_backend(args.ck_loss_backend)
    )
    clip_enabled = float(args.max_grad_norm) > 0.0
    safety = {
        "enforce_production_safety": bool(args.enforce_production_safety),
        "allow_unsafe_adamw_lr": bool(args.allow_unsafe_adamw_lr),
        "unsafe_adamw_lr_threshold": float(args.unsafe_adamw_lr_threshold),
        "risky_all_c_adamw": bool(risky_all_c_adamw),
        "max_grad_norm": float(args.max_grad_norm),
        "grad_clip_configured": bool(clip_enabled),
        "status": "ok",
        "message": "",
    }
    if risky_all_c_adamw:
        safety["status"] = "unsafe"
        safety["message"] = (
            "all-C AdamW long-horizon with lr >= threshold is high-risk "
            "(known drift around step ~800 at lr=1e-3). "
            "Production path should lower --lr below threshold. "
            "Use --allow-unsafe-adamw-lr only for diagnostics."
        )
        if args.enforce_production_safety and not args.allow_unsafe_adamw_lr:
            print(f"ERROR: {safety['message']}", file=sys.stderr)
            return 2
        if args.allow_unsafe_adamw_lr:
            safety["status"] = "unsafe_allowed"
            safety["message"] = "Unsafe AdamW LR profile explicitly allowed by CLI flag."
        print(f"WARNING: {safety['message']}", file=sys.stderr)
    init_state = None
    if (args.weights_bump is None) != (args.weights_manifest is None):
        print("ERROR: --weights-bump and --weights-manifest must be provided together", file=sys.stderr)
        return 2
    if args.weights_bump is not None:
        if not args.weights_bump.exists():
            print(f"ERROR: weights.bump not found: {args.weights_bump}", file=sys.stderr)
            return 2
        if not args.weights_manifest.exists():
            print(f"ERROR: weights_manifest.json not found: {args.weights_manifest}", file=sys.stderr)
            return 2
        try:
            init_state, dims = _load_tiny_state_from_bump(args.weights_bump, args.weights_manifest)
        except Exception as e:
            print(f"ERROR: failed to load tiny init from bump: {e}", file=sys.stderr)
            print("Hint: regenerate run_dir with cks-v7-run init (updated v7 init format).", file=sys.stderr)
            return 2
        args.vocab = int(dims["vocab"])
        args.d_model = int(dims["d_model"])
        args.hidden = int(dims["hidden"])
        print(
            f"Loaded tiny parity init from bump: vocab={args.vocab} d_model={args.d_model} hidden={args.hidden}",
            file=sys.stderr,
        )

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
        train_text=args.train_text,
        init_state=init_state,
        max_steps=args.max_steps if args.max_steps > 0 else None,
        diag_every=args.diag_every,
        ck_rmsnorm_backend=args.ck_rmsnorm_backend,
        ck_swiglu_backend=args.ck_swiglu_backend,
        ck_loss_backend=args.ck_loss_backend,
        drift_localize_step=args.drift_localize_step,
        drift_localize_tol=args.drift_localize_tol,
        drift_localize_source=args.drift_localize_source,
        epoch_snapshot_every=args.epoch_snapshot_every,
        epoch_snapshot_topk=args.epoch_snapshot_topk,
        max_grad_norm=args.max_grad_norm,
        safety=safety,
    )

    print("=" * 100)
    print("v7 TRAIN PARITY (multi-epoch)")
    print("=" * 100)
    print(f"epochs={stats.epochs} seq_len={stats.seq_len} total_tokens={stats.total_tokens} "
          f"grad_accum={stats.grad_accum} optimizer={stats.optimizer} lr={stats.lr}")
    print(
        f"ck-kernel-backends: rmsnorm={args.ck_rmsnorm_backend} "
        f"swiglu={args.ck_swiglu_backend} loss={args.ck_loss_backend}"
    )
    print(f"micro_steps={stats.micro_steps} optimizer_steps={stats.steps} tokens_per_update={stats.tokens_per_update}")
    print(f"max_grad_norm={float(args.max_grad_norm):.6g} safety_status={stats.safety.get('status', 'ok')}")
    print(f"max_loss_abs_diff={stats.max_loss_abs_diff:.3e} mean_loss_abs_diff={stats.mean_loss_abs_diff:.3e}")
    print(f"final_ck_loss={stats.final_ck_loss:.6f} final_torch_loss={stats.final_torch_loss:.6f}")
    print(f"final_param_max_abs_diff={stats.final_param_max_abs_diff:.3e} "
          f"final_param_mean_abs_diff={stats.final_param_mean_abs_diff:.3e}")
    if stats.drift_diagnostics:
        print(
            "drift: "
            f"first_loss_fail_step={stats.drift_diagnostics.get('first_loss_fail_step')} "
            f"first_param_fail_step={stats.drift_diagnostics.get('first_param_fail_step')} "
            f"max_logit_abs_diff={stats.drift_diagnostics.get('max_logit_abs_diff', 0.0):.3e} "
            f"max_grad_abs_diff={stats.drift_diagnostics.get('max_grad_abs_diff', 0.0):.3e}"
        )
        localize = stats.drift_diagnostics.get("localize_step_report")
        if isinstance(localize, dict) and localize:
            stage_max = localize.get("stage_max_global") if isinstance(localize.get("stage_max_global"), dict) else {}
            top_stage = None
            top_val = -1.0
            for k, v in stage_max.items():
                fv = float(v)
                if fv > top_val:
                    top_stage = k
                    top_val = fv
            print(
                "localize: "
                f"step={localize.get('target_step')} "
                f"source={localize.get('same_state_source')} "
                f"first_stage_over_tol={localize.get('first_stage_over_tol')} "
                f"top_stage={top_stage}({max(top_val, 0.0):.3e})"
            )
    print("PARITY:", "PASS" if stats.pass_parity else "FAIL")
    print("=" * 100)

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(stats.__dict__, indent=2), encoding="utf-8")
        print(f"JSON: {args.json_out}")

    return 0 if stats.pass_parity else 1


if __name__ == "__main__":
    raise SystemExit(main())

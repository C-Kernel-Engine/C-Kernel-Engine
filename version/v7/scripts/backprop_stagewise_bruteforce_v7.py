#!/usr/bin/env python3
"""
backprop_stagewise_bruteforce_v7.py

Methodical CK-vs-PyTorch backprop drift finder:
1) Single-window stagewise probe (forward + backward) at seq_len=1.
2) Multi-epoch trajectory sync check at seq_len=1.
3) Repeat for seq_len=2 with configurable grad_accum to stress accumulation.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[2]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import train_parity_epochs_v7 as tp  # noqa: E402


def _build_stage_tensors_stacked_ck(
    model: tp.StackedCKModel,
    x: torch.Tensor,
    targets: torch.Tensor,
    lib,
    loss_backend: str,
) -> Tuple[List[str], Dict[str, torch.Tensor]]:
    stages: Dict[str, torch.Tensor] = {}
    order: List[str] = []

    h = model.embedding(x).reshape(-1, model.d_model)
    stages["embedding"] = h
    order.append("embedding")

    for li, blk in enumerate(model.blocks):
        residual = h
        if model.use_c_rmsnorm:
            r = tp.c_rmsnorm(h, blk.rms_gamma, model.eps, model.lib)
        else:
            r = tp._rmsnorm_apply_torch(h, blk.rms_gamma, model.eps)
        stages[f"layer{li}.rmsnorm"] = r
        order.append(f"layer{li}.rmsnorm")

        f1 = blk.fc1(r)
        stages[f"layer{li}.fc1"] = f1
        order.append(f"layer{li}.fc1")

        if model.use_c_swiglu:
            sw = tp.c_swiglu(f1, model.lib)
        else:
            sw = tp._swiglu_apply_torch(f1, model.hidden)
        stages[f"layer{li}.swiglu"] = sw
        order.append(f"layer{li}.swiglu")

        f2 = blk.fc2(sw)
        stages[f"layer{li}.fc2"] = f2
        order.append(f"layer{li}.fc2")

        h = residual + f2
        stages[f"layer{li}.residual_add"] = h
        order.append(f"layer{li}.residual_add")

    if model.use_c_rmsnorm:
        final_norm = tp.c_rmsnorm(h, model.final_ln_gamma, model.eps, lib)
    else:
        final_norm = tp._rmsnorm_apply_torch(h, model.final_ln_gamma, model.eps)
    stages["final.rmsnorm"] = final_norm
    order.append("final.rmsnorm")

    logits = model.lm_head(final_norm)
    stages["logits"] = logits
    order.append("logits")

    if tp._is_c_loss_backend(loss_backend):
        loss = tp.c_cross_entropy(
            logits,
            targets,
            lib,
            kernel_variant=tp._c_loss_kernel_variant(loss_backend),
        )
    else:
        loss = F.cross_entropy(logits, targets, reduction="mean")
    stages["loss"] = loss
    order.append("loss")

    return order, stages


def _build_stage_tensors_stacked_torch(
    model: tp.StackedTorchModel,
    x: torch.Tensor,
    targets: torch.Tensor,
) -> Tuple[List[str], Dict[str, torch.Tensor]]:
    stages: Dict[str, torch.Tensor] = {}
    order: List[str] = []

    h = model.embedding(x).reshape(-1, model.d_model)
    stages["embedding"] = h
    order.append("embedding")

    for li, blk in enumerate(model.blocks):
        residual = h
        r = tp._rmsnorm_apply_torch(h, blk.rms_gamma, model.eps)
        stages[f"layer{li}.rmsnorm"] = r
        order.append(f"layer{li}.rmsnorm")

        f1 = blk.fc1(r)
        stages[f"layer{li}.fc1"] = f1
        order.append(f"layer{li}.fc1")

        sw = tp._swiglu_apply_torch(f1, model.hidden)
        stages[f"layer{li}.swiglu"] = sw
        order.append(f"layer{li}.swiglu")

        f2 = blk.fc2(sw)
        stages[f"layer{li}.fc2"] = f2
        order.append(f"layer{li}.fc2")

        h = residual + f2
        stages[f"layer{li}.residual_add"] = h
        order.append(f"layer{li}.residual_add")

    final_norm = tp._rmsnorm_apply_torch(h, model.final_ln_gamma, model.eps)
    stages["final.rmsnorm"] = final_norm
    order.append("final.rmsnorm")

    logits = model.lm_head(final_norm)
    stages["logits"] = logits
    order.append("logits")

    loss = F.cross_entropy(logits, targets, reduction="mean")
    stages["loss"] = loss
    order.append("loss")

    return order, stages


def _max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    d = (a.detach() - b.detach()).abs()
    return float(d.max().item()) if d.numel() else 0.0


def _ce_formula_grad(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    n = max(1, int(targets.numel()))
    probs = torch.softmax(logits.detach(), dim=-1)
    one_hot = torch.zeros_like(probs)
    one_hot.scatter_(1, targets.reshape(-1, 1), 1.0)
    return (probs - one_hot) / float(n)


def _first_stage(stage_diffs: Dict[str, float], order: List[str], tol: float) -> Dict[str, object]:
    first_nonzero = "none"
    first_over_tol = "none"
    for name in order:
        v = float(stage_diffs.get(name, 0.0))
        if first_nonzero == "none" and v > 0.0:
            first_nonzero = name
        if first_over_tol == "none" and v > float(tol):
            first_over_tol = name
    return {
        "first_nonzero": first_nonzero,
        "first_over_tol": first_over_tol,
    }


def _make_models(
    lib,
    *,
    vocab: int,
    d_model: int,
    hidden: int,
    eps: float,
    num_layers: int,
    ck_rmsnorm_backend: str,
    ck_swiglu_backend: str,
    init_state: Dict[str, torch.Tensor] | None,
) -> Tuple[tp.StackedCKModel, tp.StackedTorchModel]:
    ck = tp.StackedCKModel(
        vocab=vocab,
        d_model=d_model,
        hidden=hidden,
        eps=eps,
        lib=lib,
        num_layers=num_layers,
        use_c_rmsnorm=(str(ck_rmsnorm_backend).lower() == "c"),
        use_c_swiglu=(str(ck_swiglu_backend).lower() == "c"),
    )
    t = tp.StackedTorchModel(
        vocab=vocab,
        d_model=d_model,
        hidden=hidden,
        eps=eps,
        num_layers=num_layers,
    )
    if init_state is not None:
        ck.load_state_dict(init_state, strict=True)
        t.load_state_dict(init_state, strict=True)
    else:
        t.load_state_dict(ck.state_dict(), strict=True)
    return ck, t


def run_stage_probe(
    *,
    lib,
    seq_len: int,
    grad_accum: int,
    vocab: int,
    d_model: int,
    hidden: int,
    num_layers: int,
    eps: float,
    lr: float,
    seed: int,
    total_tokens: int,
    train_text: str | None,
    ck_rmsnorm_backend: str,
    ck_swiglu_backend: str,
    ck_loss_backend: str,
    drift_tol: float,
    init_state: Dict[str, torch.Tensor] | None,
) -> Dict[str, object]:
    tp._seed_all(seed)
    ck, t = _make_models(
        lib,
        vocab=vocab,
        d_model=d_model,
        hidden=hidden,
        eps=eps,
        num_layers=num_layers,
        ck_rmsnorm_backend=ck_rmsnorm_backend,
        ck_swiglu_backend=ck_swiglu_backend,
        init_state=init_state,
    )
    ck.train()
    t.train()

    opt_ck = tp._make_optimizer("adamw", ck.parameters(), lr=lr)
    opt_t = tp._make_optimizer("adamw", t.parameters(), lr=lr)
    opt_ck.zero_grad(set_to_none=True)
    opt_t.zero_grad(set_to_none=True)

    if train_text:
        batches = tp._build_batches_from_text(train_text, total_tokens=total_tokens, seq_len=seq_len, vocab=vocab)
    else:
        batches = tp._build_batches(total_tokens=total_tokens, seq_len=seq_len, vocab=vocab, seed=seed + 17)
    if not batches:
        raise RuntimeError("no batches built for stage probe")

    fwd_order: List[str] = []
    fwd_max: Dict[str, float] = {}
    bwd_order: List[str] = []
    bwd_max: Dict[str, float] = {}
    ce_rows: List[Dict[str, float]] = []

    micro_done = 0
    for x, y in batches:
        targets = y.reshape(-1)
        o_ck, s_ck = _build_stage_tensors_stacked_ck(ck, x, targets, lib, ck_loss_backend)
        o_t, s_t = _build_stage_tensors_stacked_torch(t, x, targets)
        if not fwd_order:
            fwd_order = list(o_ck)
            bwd_order = [n for n in reversed(o_ck) if n != "loss"]
            for n in fwd_order:
                fwd_max[n] = 0.0
            for n in bwd_order:
                bwd_max[n] = 0.0

        for name in fwd_order:
            d = _max_abs_diff(s_ck[name], s_t[name])
            fwd_max[name] = max(float(fwd_max.get(name, 0.0)), float(d))
            if s_ck[name].requires_grad:
                s_ck[name].retain_grad()
            if s_t[name].requires_grad:
                s_t[name].retain_grad()

        (s_ck["loss"] / float(grad_accum)).backward()
        (s_t["loss"] / float(grad_accum)).backward()

        for name in bwd_order:
            g_ck = s_ck[name].grad
            g_t = s_t[name].grad
            if g_ck is None and g_t is None:
                d = 0.0
            elif g_ck is None or g_t is None:
                d = float("inf")
            else:
                d = _max_abs_diff(g_ck, g_t)
            bwd_max[name] = max(float(bwd_max.get(name, 0.0)), float(d))

        lg_ck = s_ck["logits"].grad
        lg_t = s_t["logits"].grad
        if lg_ck is not None and lg_t is not None:
            formula_ck = _ce_formula_grad(s_ck["logits"], targets)
            formula_t = _ce_formula_grad(s_t["logits"], targets)
            ce_rows.append(
                {
                    "micro_step": int(micro_done + 1),
                    "dlogits_ck_vs_torch": _max_abs_diff(lg_ck, lg_t),
                    "dlogits_ck_vs_formula": _max_abs_diff(lg_ck, formula_ck),
                    "dlogits_torch_vs_formula": _max_abs_diff(lg_t, formula_t),
                }
            )

        micro_done += 1
        if micro_done >= grad_accum:
            break

    if micro_done == 0:
        raise RuntimeError("stage probe completed zero micro-steps")

    grad_diffs = {}
    ck_named = {n: p for n, p in ck.named_parameters()}
    t_named = {n: p for n, p in t.named_parameters()}
    watch_params = [
        "embedding.weight",
        "blocks.0.rms_gamma",
        "final_ln_gamma",
        "blocks.0.fc1.weight",
        "blocks.0.fc2.weight",
        "lm_head.weight",
    ]
    for name in watch_params:
        p_ck = ck_named.get(name)
        p_t = t_named.get(name)
        if p_ck is None or p_t is None:
            continue
        if p_ck.grad is None or p_t.grad is None:
            grad_diffs[name] = {"missing_grad": True}
            continue
        grad_diffs[name] = {
            "max_abs_diff": _max_abs_diff(p_ck.grad, p_t.grad),
            "ck_grad_max_abs": float(p_ck.grad.detach().abs().max().item()) if p_ck.grad.numel() else 0.0,
            "torch_grad_max_abs": float(p_t.grad.detach().abs().max().item()) if p_t.grad.numel() else 0.0,
        }

    pre_param_max, pre_param_mean, pre_worst = tp._state_dict_diff_stats(ck.state_dict(), t.state_dict())
    pre_opt = tp._optimizer_state_diff_stats(ck, t, opt_ck, opt_t)
    opt_ck.step()
    opt_t.step()
    opt_ck.zero_grad(set_to_none=True)
    opt_t.zero_grad(set_to_none=True)
    post_param_max, post_param_mean, post_worst = tp._state_dict_diff_stats(ck.state_dict(), t.state_dict())
    post_opt = tp._optimizer_state_diff_stats(ck, t, opt_ck, opt_t)

    fwd_first = _first_stage(fwd_max, fwd_order, drift_tol)
    bwd_first = _first_stage(bwd_max, bwd_order, drift_tol)

    return {
        "seq_len": int(seq_len),
        "grad_accum": int(grad_accum),
        "micro_steps_checked": int(micro_done),
        "drift_tol": float(drift_tol),
        "forward_stage_order": fwd_order,
        "forward_stage_max_abs_diff": fwd_max,
        "backward_stage_order": bwd_order,
        "backward_stage_max_abs_diff": bwd_max,
        "first_forward_drift": fwd_first,
        "first_backward_drift": bwd_first,
        "ce_p_minus_one_hot_checks": ce_rows,
        "watched_param_grad_diffs": grad_diffs,
        "pre_step_same_state": {
            "max_param_diff": float(pre_param_max),
            "mean_param_diff": float(pre_param_mean),
            "worst_param": pre_worst,
            **pre_opt,
        },
        "post_step_same_state": {
            "max_param_diff": float(post_param_max),
            "mean_param_diff": float(post_param_mean),
            "worst_param": post_worst,
            **post_opt,
        },
    }


def run_sync_check(
    *,
    lib,
    seq_len: int,
    grad_accum: int,
    vocab: int,
    d_model: int,
    hidden: int,
    num_layers: int,
    eps: float,
    lr: float,
    seed: int,
    total_tokens: int,
    epochs: int,
    train_text: str | None,
    ck_rmsnorm_backend: str,
    ck_swiglu_backend: str,
    ck_loss_backend: str,
    init_state: Dict[str, torch.Tensor] | None,
) -> Dict[str, object]:
    stats = tp.run_training_parity(
        lib=lib,
        model_kind="stacked",
        num_layers=num_layers,
        epochs=epochs,
        seq_len=seq_len,
        total_tokens=total_tokens,
        vocab=vocab,
        d_model=d_model,
        hidden=hidden,
        eps=eps,
        grad_accum=grad_accum,
        optimizer="adamw",
        lr=lr,
        seed=seed,
        loss_tol=2e-5,
        param_tol=3e-5,
        train_text=train_text,
        init_state=copy.deepcopy(init_state) if init_state is not None else None,
        max_steps=None,
        diag_every=1,
        ck_rmsnorm_backend=ck_rmsnorm_backend,
        ck_swiglu_backend=ck_swiglu_backend,
        ck_loss_backend=ck_loss_backend,
        drift_localize_step=0,
        max_grad_norm=0.0,
        safety={"status": "bruteforce"},
    )
    out = asdict(stats)
    return {
        "pass_parity": bool(out["pass_parity"]),
        "steps": int(out["steps"]),
        "micro_steps": int(out["micro_steps"]),
        "max_loss_abs_diff": float(out["max_loss_abs_diff"]),
        "final_param_max_abs_diff": float(out["final_param_max_abs_diff"]),
        "drift_diagnostics": out.get("drift_diagnostics", {}),
        "final_ck_loss": float(out["final_ck_loss"]),
        "final_torch_loss": float(out["final_torch_loss"]),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Stagewise brute-force backprop parity drill (seq_len=1 then 2).")
    ap.add_argument("--seq1-len", type=int, default=1)
    ap.add_argument("--seq2-len", type=int, default=2)
    ap.add_argument("--seq1-grad-accum", type=int, default=1)
    ap.add_argument("--seq2-grad-accum", type=int, default=4)
    ap.add_argument("--sync-epochs", type=int, default=10)
    ap.add_argument("--sync-total-tokens", type=int, default=256)
    ap.add_argument("--probe-total-tokens", type=int, default=64)
    ap.add_argument("--vocab", type=int, default=256)
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--num-layers", type=int, default=1)
    ap.add_argument("--eps", type=float, default=1e-5)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--drift-tol", type=float, default=1e-8)
    ap.add_argument("--train-text", type=str, default="Hello!")
    ap.add_argument("--ck-rmsnorm-backend", choices=["c", "torch"], default="c")
    ap.add_argument("--ck-swiglu-backend", choices=["c", "torch"], default="c")
    ap.add_argument("--ck-loss-backend", choices=["c", "c_ptref", "torch"], default="c")
    ap.add_argument("--weights-bump", type=Path, default=None)
    ap.add_argument("--weights-manifest", type=Path, default=None)
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args()

    if args.seq1_len < 1 or args.seq2_len < 1:
        print("ERROR: seq lens must be >= 1", file=sys.stderr)
        return 2
    if args.seq1_grad_accum < 1 or args.seq2_grad_accum < 1:
        print("ERROR: grad_accum must be >= 1", file=sys.stderr)
        return 2
    if args.num_layers < 1:
        print("ERROR: --num-layers must be >= 1", file=sys.stderr)
        return 2
    if args.sync_epochs < 1:
        print("ERROR: --sync-epochs must be >= 1", file=sys.stderr)
        return 2
    if (args.weights_bump is None) != (args.weights_manifest is None):
        print("ERROR: provide both --weights-bump and --weights-manifest together", file=sys.stderr)
        return 2

    lib = tp._load_lib()
    init_state = None
    vocab = int(args.vocab)
    d_model = int(args.d_model)
    hidden = int(args.hidden)
    num_layers = int(args.num_layers)
    if args.weights_bump is not None:
        state, dims, kind = tp._load_init_state_from_bump(
            args.weights_bump,
            args.weights_manifest,
            model_kind="stacked",
        )
        if str(kind) != "stacked":
            print(f"ERROR: expected stacked init, got: {kind}", file=sys.stderr)
            return 2
        init_state = state
        vocab = int(dims["vocab"])
        d_model = int(dims["d_model"])
        hidden = int(dims["hidden"])
        num_layers = int(dims["num_layers"])

    seq1_probe = run_stage_probe(
        lib=lib,
        seq_len=int(args.seq1_len),
        grad_accum=int(args.seq1_grad_accum),
        vocab=vocab,
        d_model=d_model,
        hidden=hidden,
        num_layers=num_layers,
        eps=float(args.eps),
        lr=float(args.lr),
        seed=int(args.seed),
        total_tokens=max(int(args.probe_total_tokens), int(args.seq1_len) + 1),
        train_text=args.train_text,
        ck_rmsnorm_backend=args.ck_rmsnorm_backend,
        ck_swiglu_backend=args.ck_swiglu_backend,
        ck_loss_backend=args.ck_loss_backend,
        drift_tol=float(args.drift_tol),
        init_state=copy.deepcopy(init_state) if init_state is not None else None,
    )
    seq1_sync = run_sync_check(
        lib=lib,
        seq_len=int(args.seq1_len),
        grad_accum=int(args.seq1_grad_accum),
        vocab=vocab,
        d_model=d_model,
        hidden=hidden,
        num_layers=num_layers,
        eps=float(args.eps),
        lr=float(args.lr),
        seed=int(args.seed),
        total_tokens=max(int(args.sync_total_tokens), int(args.seq1_len) + 1),
        epochs=int(args.sync_epochs),
        train_text=args.train_text,
        ck_rmsnorm_backend=args.ck_rmsnorm_backend,
        ck_swiglu_backend=args.ck_swiglu_backend,
        ck_loss_backend=args.ck_loss_backend,
        init_state=copy.deepcopy(init_state) if init_state is not None else None,
    )

    seq2_probe = run_stage_probe(
        lib=lib,
        seq_len=int(args.seq2_len),
        grad_accum=int(args.seq2_grad_accum),
        vocab=vocab,
        d_model=d_model,
        hidden=hidden,
        num_layers=num_layers,
        eps=float(args.eps),
        lr=float(args.lr),
        seed=int(args.seed),
        total_tokens=max(int(args.probe_total_tokens), int(args.seq2_len) * int(args.seq2_grad_accum) + 1),
        train_text=args.train_text,
        ck_rmsnorm_backend=args.ck_rmsnorm_backend,
        ck_swiglu_backend=args.ck_swiglu_backend,
        ck_loss_backend=args.ck_loss_backend,
        drift_tol=float(args.drift_tol),
        init_state=copy.deepcopy(init_state) if init_state is not None else None,
    )
    seq2_sync = run_sync_check(
        lib=lib,
        seq_len=int(args.seq2_len),
        grad_accum=int(args.seq2_grad_accum),
        vocab=vocab,
        d_model=d_model,
        hidden=hidden,
        num_layers=num_layers,
        eps=float(args.eps),
        lr=float(args.lr),
        seed=int(args.seed),
        total_tokens=max(int(args.sync_total_tokens), int(args.seq2_len) + 1),
        epochs=int(args.sync_epochs),
        train_text=args.train_text,
        ck_rmsnorm_backend=args.ck_rmsnorm_backend,
        ck_swiglu_backend=args.ck_swiglu_backend,
        ck_loss_backend=args.ck_loss_backend,
        init_state=copy.deepcopy(init_state) if init_state is not None else None,
    )

    payload = {
        "config": {
            "vocab": vocab,
            "d_model": d_model,
            "hidden": hidden,
            "num_layers": num_layers,
            "eps": float(args.eps),
            "lr": float(args.lr),
            "seed": int(args.seed),
            "train_text": args.train_text,
            "ck_backends": {
                "rmsnorm": str(args.ck_rmsnorm_backend),
                "swiglu": str(args.ck_swiglu_backend),
                "loss": str(args.ck_loss_backend),
            },
        },
        "seq1": {"probe": seq1_probe, "sync": seq1_sync},
        "seq2": {"probe": seq2_probe, "sync": seq2_sync},
    }

    def _fmt(stage_blob: Dict[str, object]) -> str:
        f = stage_blob["first_forward_drift"]
        b = stage_blob["first_backward_drift"]
        return (
            f"first_fwd_nonzero={f['first_nonzero']} first_fwd_over_tol={f['first_over_tol']} "
            f"first_bwd_nonzero={b['first_nonzero']} first_bwd_over_tol={b['first_over_tol']}"
        )

    print("=" * 100)
    print("BACKPROP STAGEWISE BRUTEFORCE")
    print("=" * 100)
    print(f"seq1 probe: {_fmt(seq1_probe)}")
    print(
        "seq1 ce: "
        f"max_ck_vs_torch={max((r['dlogits_ck_vs_torch'] for r in seq1_probe['ce_p_minus_one_hot_checks']), default=0.0):.3e} "
        f"max_ck_vs_formula={max((r['dlogits_ck_vs_formula'] for r in seq1_probe['ce_p_minus_one_hot_checks']), default=0.0):.3e}"
    )
    print(
        "seq1 sync: "
        f"pass={seq1_sync['pass_parity']} steps={seq1_sync['steps']} "
        f"max_loss_abs_diff={seq1_sync['max_loss_abs_diff']:.3e} "
        f"final_param_max_abs_diff={seq1_sync['final_param_max_abs_diff']:.3e}"
    )
    print("-" * 100)
    print(f"seq2 probe: {_fmt(seq2_probe)}")
    print(
        "seq2 ce: "
        f"max_ck_vs_torch={max((r['dlogits_ck_vs_torch'] for r in seq2_probe['ce_p_minus_one_hot_checks']), default=0.0):.3e} "
        f"max_ck_vs_formula={max((r['dlogits_ck_vs_formula'] for r in seq2_probe['ce_p_minus_one_hot_checks']), default=0.0):.3e}"
    )
    print(
        "seq2 sync: "
        f"pass={seq2_sync['pass_parity']} steps={seq2_sync['steps']} "
        f"max_loss_abs_diff={seq2_sync['max_loss_abs_diff']:.3e} "
        f"final_param_max_abs_diff={seq2_sync['final_param_max_abs_diff']:.3e}"
    )
    print("=" * 100)

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"JSON: {args.json_out}")

    # Keep non-zero exit only when obvious sync/parity breaks appear in either run.
    overall_ok = bool(seq1_sync["pass_parity"] and seq2_sync["pass_parity"])
    return 0 if overall_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())


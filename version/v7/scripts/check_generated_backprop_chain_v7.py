#!/usr/bin/env python3
"""
check_generated_backprop_chain_v7.py

Generated-runtime (backend=ck) step-1 backprop chain comparator:
- Reads CK accum snapshot dump (grad_weights + grad_activations)
- Rebuilds equivalent Torch model from run-dir weights
- Runs one backward step on the same first token pair
- Compares selected backprop chain slots (logits -> ... -> embedding)

Intended usage:
1) Run ck generated training with one checked step and dump-on-check:
   ck_run_v7.py train ... --parity-on --parity-every 1 --dump-on-check
2) Run this script against that run-dir/report.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from oracle_snapshot_torch_v7 import _apply_affine, _apply_head_rms_norm, _apply_rope, _decode_weight_snapshot  # noqa: E402
from train_qwen3_torch_from_run_v7 import TorchQwenFromRun, _load_run_weights  # noqa: E402


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


_LAYER_ACT_RE = re.compile(r"\.L(\d+)\.")
_LAYER_W_RE = re.compile(r"^grad\.weight\.layer\.(\d+)\.")


def _layer_from_act_slot(name: str) -> int | None:
    m = _LAYER_ACT_RE.search(name)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def _layer_from_weight_slot(name: str) -> int | None:
    m = _LAYER_W_RE.search(name)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def _build_per_layer_summary(
    act_rows: List[dict],
    weight_rows: List[dict],
) -> List[dict]:
    layers = set()
    for r in act_rows:
        li = _layer_from_act_slot(str(r.get("grad_slot", "")))
        if li is not None:
            layers.add(int(li))
    for r in weight_rows:
        li = _layer_from_weight_slot(str(r.get("grad_weight", "")))
        if li is not None:
            layers.add(int(li))

    out: List[dict] = []
    for li in sorted(layers, reverse=True):
        acts = [r for r in act_rows if _layer_from_act_slot(str(r.get("grad_slot", ""))) == li]
        ws = [r for r in weight_rows if _layer_from_weight_slot(str(r.get("grad_weight", ""))) == li]
        acts_sorted = sorted(acts, key=lambda r: float(r.get("max_abs_diff", 0.0)), reverse=True)
        ws_sorted = sorted(ws, key=lambda r: float(r.get("max_abs_diff", 0.0)), reverse=True)
        act_top = acts_sorted[0] if acts_sorted else None
        w_top = ws_sorted[0] if ws_sorted else None
        out.append(
            {
                "layer": int(li),
                "act_slots_compared": int(len(acts)),
                "weight_tensors_compared": int(len(ws)),
                "act_max_abs_diff": (float(act_top["max_abs_diff"]) if act_top is not None else None),
                "act_worst_slot": (str(act_top["grad_slot"]) if act_top is not None else None),
                "weight_max_abs_diff": (float(w_top["max_abs_diff"]) if w_top is not None else None),
                "weight_worst_tensor": (str(w_top["grad_weight"]) if w_top is not None else None),
            }
        )
    return out


def _resolve_step_accum_post(report: dict, step: int) -> tuple[Path, str]:
    """Resolve accum snapshot for a specific optimizer step (post-step)."""
    oracle = report.get("oracle") if isinstance(report, dict) else None
    oracle = oracle if isinstance(oracle, dict) else {}
    target_step = max(1, int(step))

    check_dumps = oracle.get("check_dump_files")
    if isinstance(check_dumps, list):
        rows = [r for r in check_dumps if isinstance(r, dict)]
        rows.sort(key=lambda r: int(r.get("step", 0) or 0))
        for row in rows:
            if int(row.get("step", 0) or 0) == target_step:
                ap = row.get("accum_post")
                if isinstance(ap, str) and ap:
                    return Path(ap).resolve(), f"oracle.check_dump_files(step={target_step}).accum_post"
        for row in rows:
            ap = row.get("accum_post")
            if isinstance(ap, str) and ap:
                return Path(ap).resolve(), "oracle.check_dump_files(first).accum_post"

    accum_files = oracle.get("accum_snapshot_files")
    if isinstance(accum_files, list) and len(accum_files) > 0:
        target_tag = f"step_{target_step:08d}_post"
        for ap in accum_files:
            if isinstance(ap, str) and (target_tag in ap):
                return Path(ap).resolve(), f"oracle.accum_snapshot_files({target_tag})"
        if len(accum_files) >= 2 and isinstance(accum_files[1], str):
            return Path(accum_files[1]).resolve(), "oracle.accum_snapshot_files(index=1)"
        if isinstance(accum_files[0], str):
            return Path(accum_files[0]).resolve(), "oracle.accum_snapshot_files(index=0)"

    raise RuntimeError(
        f"Missing step-{target_step} accum post snapshot. Run ck_run_v7.py train with --parity-on --dump-on-check."
    )


def _resolve_step_weight_pre(report: dict, step: int) -> tuple[Path, str]:
    """Resolve pre-step weight snapshot for a specific optimizer step."""
    oracle = report.get("oracle") if isinstance(report, dict) else None
    oracle = oracle if isinstance(oracle, dict) else {}
    target_step = max(1, int(step))

    check_dumps = oracle.get("check_dump_files")
    if isinstance(check_dumps, list):
        rows = [r for r in check_dumps if isinstance(r, dict)]
        rows.sort(key=lambda r: int(r.get("step", 0) or 0))
        for row in rows:
            if int(row.get("step", 0) or 0) == target_step:
                wp = row.get("weight_pre")
                if isinstance(wp, str) and wp:
                    return Path(wp).resolve(), f"oracle.check_dump_files(step={target_step}).weight_pre"
        for row in rows:
            wp = row.get("weight_pre")
            if isinstance(wp, str) and wp:
                return Path(wp).resolve(), "oracle.check_dump_files(first).weight_pre"

    snap_files = oracle.get("snapshot_files")
    if isinstance(snap_files, list) and len(snap_files) > 0:
        target_tag = f"step_{target_step:08d}_pre"
        for p in snap_files:
            if isinstance(p, str) and (target_tag in p):
                return Path(p).resolve(), f"oracle.snapshot_files({target_tag})"
        if isinstance(snap_files[0], str):
            return Path(snap_files[0]).resolve(), "oracle.snapshot_files(index=0)"

    raise RuntimeError(
        f"Missing step-{target_step} weight_pre snapshot. Run ck_run_v7.py train with --parity-on --dump-on-check."
    )


def _decode_ck_accum_sections(
    runtime_summary: dict,
    accum_post_bin: Path,
) -> tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    blob = np.frombuffer(accum_post_bin.read_bytes(), dtype=np.float32)
    slots = runtime_summary.get("tensor_slots") or []
    grad_w = sorted(
        [s for s in slots if isinstance(s, dict) and s.get("section") == "grad_weights"],
        key=lambda r: (int(r.get("offset", 0) or 0), str(r.get("name", ""))),
    )
    grad_a = sorted(
        [s for s in slots if isinstance(s, dict) and s.get("section") == "grad_activations"],
        key=lambda r: (int(r.get("offset", 0) or 0), str(r.get("name", ""))),
    )
    n_w = int(sum(int(s.get("numel", 0) or 0) for s in grad_w))
    n_a = int(sum(int(s.get("numel", 0) or 0) for s in grad_a))
    if blob.size < (n_w + n_a):
        raise RuntimeError(
            f"Accum snapshot too small: got={blob.size}, expected>={n_w + n_a}"
        )
    out_w: Dict[str, np.ndarray] = {}
    out_a: Dict[str, np.ndarray] = {}
    cursor = 0
    for s in grad_w:
        name = str(s.get("name", ""))
        n = int(s.get("numel", 0) or 0)
        out_w[name] = blob[cursor : cursor + n].copy()
        cursor += n
    for s in grad_a:
        name = str(s.get("name", ""))
        n = int(s.get("numel", 0) or 0)
        out_a[name] = blob[cursor : cursor + n].copy()
        cursor += n
    return out_w, out_a


def _param_name_to_grad_weight_slot(name: str) -> str | None:
    if name == "token_emb":
        return "grad.weight.token_emb"
    if name == "final_ln_weight":
        return "grad.weight.final_ln_weight"
    if name == "output_weight":
        return "grad.weight.output.weight"
    if not name.startswith("layers."):
        return None
    parts = name.split(".")
    # layers.{i}.{param}
    if len(parts) != 3:
        return None
    if not parts[1].isdigit():
        return None
    return f"grad.weight.layer.{parts[1]}.{parts[2]}"


def _capture_torch_activations_and_backward(
    run_dir: Path,
    runtime_summary: dict,
    x_ids: list[int],
    y_ids: list[int],
    weight_snapshot_pre: Path | None = None,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, object]]:
    if weight_snapshot_pre is not None:
        snap = np.frombuffer(weight_snapshot_pre.read_bytes(), dtype=np.float32)
        tensors, cfg = _decode_weight_snapshot(run_dir, runtime_summary, snap)
        model = TorchQwenFromRun(tensors, cfg)
    else:
        loaded = _load_run_weights(run_dir)
        model = TorchQwenFromRun(loaded.tensors, loaded.cfg)
    model.train()

    if len(x_ids) != len(y_ids):
        raise ValueError(f"x/y length mismatch: {len(x_ids)} vs {len(y_ids)}")
    if len(x_ids) <= 0:
        raise ValueError("Need at least one token for backward capture")

    x = torch.tensor([[int(v) for v in x_ids]], dtype=torch.long)
    y = torch.tensor([[int(v) for v in y_ids]], dtype=torch.long)

    acts: Dict[str, torch.Tensor] = {}

    def cap(name: str, t: torch.Tensor) -> torch.Tensor:
        t.retain_grad()
        acts[name] = t
        return t

    xv = F.embedding(x, model.token_emb)
    cap("act.Sheader.dense_embedding_lookup.0.out", xv)

    for li, layer in enumerate(model.layers):
        sp = f"act.L{li}."
        residual = xv

        var1 = torch.mean(xv * xv, dim=-1, keepdim=True)
        x_norm = xv * torch.rsqrt(var1 + model.eps) * layer.ln1_gamma
        cap(sp + "rmsnorm.0.output", x_norm)

        q = _apply_affine(x_norm, layer.wq, layer.bq)
        k = _apply_affine(x_norm, layer.wk, layer.bk)
        v = _apply_affine(x_norm, layer.wv, layer.bv)
        q_expected = max(1, int(model.num_heads) * int(model.head_dim))
        kv_expected = max(1, int(model.num_kv_heads) * int(model.head_dim))
        if int(q.shape[-1]) >= q_expected:
            q = q[..., :q_expected]
        if int(k.shape[-1]) >= kv_expected:
            k = k[..., :kv_expected]
        if int(v.shape[-1]) >= kv_expected:
            v = v[..., :kv_expected]
        cap(sp + "q_proj.0.y", q)
        cap(sp + "k_proj.0.y", k)
        cap(sp + "v_proj.0.y", v)

        bsz, tsz, _ = q.shape
        q_out = int(q.shape[-1])
        k_out = int(k.shape[-1])
        q_heads = (
            model.num_heads
            if (model.num_heads * model.head_dim) == q_out
            else max(1, q_out // max(1, model.head_dim))
        )
        kv_heads_nom = (
            model.num_kv_heads
            if (model.num_kv_heads * model.head_dim) == k_out
            else max(1, k_out // max(1, model.head_dim))
        )

        qh = q.view(bsz, tsz, q_heads, model.head_dim).permute(0, 2, 1, 3).contiguous()
        kh = k.view(bsz, tsz, kv_heads_nom, model.head_dim).permute(0, 2, 1, 3).contiguous()
        vh = v.view(bsz, tsz, kv_heads_nom, model.head_dim).permute(0, 2, 1, 3).contiguous()

        qh = _apply_head_rms_norm(qh, layer.q_norm, model.eps)
        kh = _apply_head_rms_norm(kh, layer.k_norm, model.eps)
        cap(sp + "qk_norm.0.q", qh.permute(0, 2, 1, 3).contiguous().view(bsz, tsz, -1))
        cap(sp + "qk_norm.0.k", kh.permute(0, 2, 1, 3).contiguous().view(bsz, tsz, -1))

        qh, kh = _apply_rope(qh, kh, model.rope_theta)
        scale = 1.0 / math.sqrt(float(max(1, model.head_dim)))
        hq = int(qh.shape[1])
        hk = int(kh.shape[1])
        kh_eff = kh
        vh_eff = vh
        if hq != hk:
            rep = hq // hk
            kh_eff = kh.repeat_interleave(rep, dim=1)
            vh_eff = vh.repeat_interleave(rep, dim=1)

        scores = torch.matmul(qh, kh_eff.transpose(-2, -1)) * float(scale)
        t = int(scores.shape[-1])
        mask = torch.triu(torch.ones((t, t), dtype=torch.bool, device=scores.device), diagonal=1)
        scores = scores.masked_fill(mask.view(1, 1, t, t), float("-inf"))
        probs = torch.softmax(scores, dim=-1)
        ctx = torch.matmul(probs, vh_eff)
        attn_out = ctx.permute(0, 2, 1, 3).contiguous().view(bsz, tsz, -1)
        cap(sp + "attn.0.out", attn_out)

        out_proj = _apply_affine(attn_out, layer.wo, layer.bo)
        cap(sp + "out_proj.0.y", out_proj)
        xv = residual + out_proj
        cap(sp + "residual_add.0.out", xv)

        residual2 = xv
        var2 = torch.mean(xv * xv, dim=-1, keepdim=True)
        x_norm2 = xv * torch.rsqrt(var2 + model.eps) * layer.ln2_gamma
        cap(sp + "rmsnorm.1.output", x_norm2)

        gate_up = _apply_affine(x_norm2, layer.w1, layer.b1)
        cap(sp + "mlp_gate_up.0.y", gate_up)
        half = int(gate_up.shape[-1]) // 2
        gate = gate_up[..., :half]
        up = gate_up[..., half:]
        silu_mul = F.silu(gate) * up
        cap(sp + "silu_mul.0.out", silu_mul)
        mlp_down = _apply_affine(silu_mul, layer.w2, layer.b2)
        cap(sp + "mlp_down.0.y", mlp_down)
        xv = residual2 + mlp_down
        cap(sp + "residual_add.1.out", xv)

    varf = torch.mean(xv * xv, dim=-1, keepdim=True)
    footer_out = xv * torch.rsqrt(varf + model.eps) * model.final_ln_weight
    cap("act.Sfooter.rmsnorm.0.output", footer_out)
    out_w = model.output_weight if model.output_weight is not None else model.token_emb
    logits = torch.matmul(footer_out, out_w.transpose(0, 1))
    cap("act.Sfooter.logits.0.y", logits)

    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1), reduction="mean")
    loss.backward()

    act_grads: Dict[str, np.ndarray] = {}
    for k, t in acts.items():
        if t.grad is None:
            continue
        act_grads["grad.act." + k] = (
            t.grad.detach().cpu().float().reshape(-1).numpy().astype(np.float32, copy=False)
        )
    param_grads: Dict[str, np.ndarray] = {}
    for pname, p in model.named_parameters():
        if p.grad is None:
            continue
        gslot = _param_name_to_grad_weight_slot(pname)
        if gslot is None:
            continue
        param_grads[gslot] = p.grad.detach().cpu().float().reshape(-1).numpy().astype(np.float32, copy=False)
    meta = {
        "output_weight_present": bool(model.output_weight is not None),
        "weight_tied": bool(model.output_weight is None),
    }
    return act_grads, param_grads, meta


def _batch_for_step_from_text(
    text: str,
    vocab: int,
    seq_len: int,
    step: int,
    total_tokens: int,
) -> tuple[list[int], list[int]]:
    if int(seq_len) <= 0:
        raise ValueError("seq_len must be >= 1")
    if int(step) <= 0:
        raise ValueError("step must be >= 1")
    if int(total_tokens) < int(seq_len) + 1:
        raise ValueError("total_tokens must be >= seq_len + 1")
    raw = (text or "").encode("utf-8", errors="ignore")
    if len(raw) < 2:
        raise ValueError("Need text that encodes to at least 2 bytes")
    ids = [int(b) % int(vocab) for b in raw]
    needed = int(total_tokens) + 1
    repeats = (needed + len(ids) - 1) // len(ids)
    stream = (ids * repeats)[:needed]
    batch_count = max(1, (int(total_tokens) - int(seq_len)) // int(seq_len) + 1)
    idx = (int(step) - 1) % int(batch_count)
    start = idx * int(seq_len)
    x_ids = [int(v) for v in stream[start : start + int(seq_len)]]
    y_ids = [int(v) for v in stream[start + 1 : start + int(seq_len) + 1]]
    return x_ids, y_ids


def main() -> int:
    ap = argparse.ArgumentParser(description="Compare generated-C step1 backprop chain slots vs Torch.")
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--report-json", type=Path, required=True, help="ck_run_v7.py train JSON report")
    ap.add_argument("--data", type=Path, default=None, help="UTF-8 text file used for training stream")
    ap.add_argument("--prompt", type=str, default=None, help="Inline training text fallback")
    ap.add_argument("--seq-len", type=int, default=1, help="Sequence length used for first-batch torch replay")
    ap.add_argument("--step", type=int, default=1, help="Optimizer step index to compare (default: 1)")
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args()

    run_dir = args.run_dir.resolve()
    report = _load_json(args.report_json.resolve())
    runtime_summary = _load_json(run_dir / "generated_train_runtime_summary_v7.json")
    manifest = _load_json(run_dir / "weights_manifest.json")
    vocab = int((manifest.get("config") or {}).get("vocab_size", 256) or 256)

    step_target = max(1, int(args.step))
    accum_post, accum_source = _resolve_step_accum_post(report, step_target)
    weight_pre, weight_pre_source = _resolve_step_weight_pre(report, step_target)
    if not accum_post.exists():
        raise FileNotFoundError(f"Missing accum snapshot: {accum_post}")
    if not weight_pre.exists():
        raise FileNotFoundError(f"Missing weight_pre snapshot: {weight_pre}")

    if args.data is not None:
        text = args.data.read_text(encoding="utf-8")
    elif args.prompt is not None:
        text = str(args.prompt)
    else:
        raise ValueError("Provide --data or --prompt")

    report_total_tokens = int(report.get("total_tokens", 0) or 0)
    seq_len_cfg = int(report.get("seq_len", int(args.seq_len)) or int(args.seq_len))
    if seq_len_cfg <= 0:
        seq_len_cfg = int(args.seq_len)
    min_total = int(step_target) * int(seq_len_cfg)
    total_tokens_for_stream = max(report_total_tokens, min_total + 1, seq_len_cfg + 1)
    x_ids, y_ids = _batch_for_step_from_text(
        text=text,
        vocab=vocab,
        seq_len=seq_len_cfg,
        step=step_target,
        total_tokens=total_tokens_for_stream,
    )
    ck_grad_w, ck_grad_a = _decode_ck_accum_sections(runtime_summary, accum_post)
    pt_grad_a, pt_grad_w, torch_meta = _capture_torch_activations_and_backward(
        run_dir,
        runtime_summary,
        x_ids,
        y_ids,
        weight_snapshot_pre=weight_pre,
    )

    rows: List[dict] = []
    for slot, ck in ck_grad_a.items():
        pt = pt_grad_a.get(slot)
        if pt is None or pt.size != ck.size:
            continue
        d = np.abs(ck - pt)
        rows.append(
            {
                "grad_slot": slot,
                "max_abs_diff": float(d.max()) if d.size else 0.0,
                "mean_abs_diff": float(d.mean()) if d.size else 0.0,
                "ck_max_abs": float(np.max(np.abs(ck))) if ck.size else 0.0,
                "pt_max_abs": float(np.max(np.abs(pt))) if pt.size else 0.0,
            }
        )
    rows.sort(key=lambda r: float(r["max_abs_diff"]), reverse=True)

    weight_rows: List[dict] = []
    for slot, ck in ck_grad_w.items():
        pt = pt_grad_w.get(slot)
        if pt is None or pt.size != ck.size:
            continue
        d = np.abs(ck - pt)
        weight_rows.append(
            {
                "grad_weight": slot,
                "max_abs_diff": float(d.max()) if d.size else 0.0,
                "mean_abs_diff": float(d.mean()) if d.size else 0.0,
                "ck_max_abs": float(np.max(np.abs(ck))) if ck.size else 0.0,
                "pt_max_abs": float(np.max(np.abs(pt))) if pt.size else 0.0,
            }
        )
    weight_rows.sort(key=lambda r: float(r["max_abs_diff"]), reverse=True)

    chain_order = [
        "grad.act.act.Sfooter.logits.0.y",
        "grad.act.act.Sfooter.rmsnorm.0.output",
        "grad.act.act.L1.residual_add.1.out",
        "grad.act.act.L1.mlp_down.0.y",
        "grad.act.act.L1.silu_mul.0.out",
        "grad.act.act.L1.mlp_gate_up.0.y",
        "grad.act.act.L1.rmsnorm.1.output",
        "grad.act.act.L1.residual_add.0.out",
        "grad.act.act.L1.out_proj.0.y",
        "grad.act.act.L1.attn.0.out",
        "grad.act.act.L1.q_proj.0.y",
        "grad.act.act.L1.k_proj.0.y",
        "grad.act.act.L1.v_proj.0.y",
        "grad.act.act.L1.rmsnorm.0.output",
        "grad.act.act.L0.residual_add.1.out",
        "grad.act.act.L0.mlp_down.0.y",
        "grad.act.act.L0.silu_mul.0.out",
        "grad.act.act.L0.mlp_gate_up.0.y",
        "grad.act.act.L0.rmsnorm.1.output",
        "grad.act.act.L0.residual_add.0.out",
        "grad.act.act.L0.out_proj.0.y",
        "grad.act.act.L0.attn.0.out",
        "grad.act.act.L0.q_proj.0.y",
        "grad.act.act.L0.k_proj.0.y",
        "grad.act.act.L0.v_proj.0.y",
        "grad.act.act.L0.rmsnorm.0.output",
        "grad.act.act.Sheader.dense_embedding_lookup.0.out",
    ]
    row_by = {r["grad_slot"]: r for r in rows}
    chain = [row_by[s] for s in chain_order if s in row_by]

    first_big = None
    for r in chain:
        if float(r["max_abs_diff"]) > 1e-3:
            first_big = r
            break

    by_w = {r["grad_weight"]: r for r in weight_rows}
    focus_weights = []
    for name in (
        "grad.weight.token_emb",
        "grad.weight.output.weight",
        "grad.weight.final_ln_weight",
        "grad.weight.layer.0.ln1_gamma",
        "grad.weight.layer.0.ln2_gamma",
    ):
        row = by_w.get(name)
        if row is not None:
            focus_weights.append(row)
    focus_acts = []
    for name in (
        "grad.act.act.Sfooter.logits.0.y",
        "grad.act.act.Sfooter.rmsnorm.0.output",
        "grad.act.act.L0.rmsnorm.1.output",
        "grad.act.act.L0.rmsnorm.0.output",
        "grad.act.act.Sheader.dense_embedding_lookup.0.out",
    ):
        row = row_by.get(name)
        if row is not None:
            focus_acts.append(row)
    rms_rows = [r for r in rows if ".rmsnorm." in str(r.get("grad_slot", ""))]
    rms_rows.sort(key=lambda r: float(r["max_abs_diff"]), reverse=True)
    per_layer = _build_per_layer_summary(rows, weight_rows)

    act_layers = sorted(
        {li for li in (_layer_from_act_slot(str(r.get("grad_slot", ""))) for r in rows) if li is not None}
    )
    weight_layers = sorted(
        {li for li in (_layer_from_weight_slot(str(r.get("grad_weight", ""))) for r in weight_rows) if li is not None}
    )
    last_layer = None
    if act_layers:
        last_layer = int(max(act_layers))
    if weight_layers:
        wl = int(max(weight_layers))
        last_layer = wl if last_layer is None else max(last_layer, wl)

    last_layer_act = []
    last_layer_weight = []
    if last_layer is not None:
        for r in rows:
            slot = str(r.get("grad_slot", ""))
            if f".L{last_layer}." in slot:
                last_layer_act.append(r)
        for r in weight_rows:
            slot = str(r.get("grad_weight", ""))
            if slot.startswith(f"grad.weight.layer.{last_layer}."):
                last_layer_weight.append(r)
        last_layer_act.sort(key=lambda r: float(r["max_abs_diff"]), reverse=True)
        last_layer_weight.sort(key=lambda r: float(r["max_abs_diff"]), reverse=True)

    loss_curve = report.get("loss_curve") if isinstance(report.get("loss_curve"), list) else []
    step_loss = {}
    for row in loss_curve:
        if isinstance(row, dict) and int(row.get("step", 0) or 0) == step_target:
            step_loss = row
            break
    if not step_loss and loss_curve and isinstance(loss_curve[0], dict):
        step_loss = loss_curve[0]
    out = {
        "run_dir": str(run_dir),
        "report_json": str(args.report_json.resolve()),
        "accum_post_snapshot": str(accum_post),
        "accum_post_snapshot_source": str(accum_source),
        "weight_pre_snapshot": str(weight_pre),
        "weight_pre_snapshot_source": str(weight_pre_source),
        "token_window": {
            "seq_len": int(seq_len_cfg),
            "total_tokens_stream": int(total_tokens_for_stream),
            "step": int(step_target),
            "x_ids": x_ids,
            "y_ids": y_ids,
        },
        "step_loss": {
            "step": int(step_loss.get("step", step_target) or step_target),
            "ck": float(step_loss.get("loss_ck", 0.0) or 0.0),
            "pt": float(step_loss.get("loss_pt", 0.0) or 0.0),
            "abs_diff": abs(float(step_loss.get("loss_ck", 0.0) or 0.0) - float(step_loss.get("loss_pt", 0.0) or 0.0)),
        },
        "torch_weight_tying": torch_meta,
        "compared_grad_slots": int(len(rows)),
        "compared_grad_weights": int(len(weight_rows)),
        "first_big_chain_diff_gt_1e-3": first_big,
        "focus_backward_checks": {
            "activations": focus_acts,
            "weight_grads": focus_weights,
            "rmsnorm_activation_top10": rms_rows[:10],
            "last_layer": {
                "layer_index": last_layer,
                "top10_activation_grads": last_layer_act[:10],
                "top10_weight_grads": last_layer_weight[:10],
            },
        },
        "per_layer_drift_desc": per_layer,
        "chain_logits_to_embedding": chain,
        "top20_grad_slot_diffs": rows[:20],
        "top20_grad_weight_diffs": weight_rows[:20],
    }

    out_path = args.json_out.resolve() if args.json_out is not None else (run_dir / "backprop_grad_slots_step1.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(f"Wrote: {out_path}")
    print(f"Compared grad slots: {len(rows)}")
    print(f"Compared grad weights: {len(weight_rows)}")
    dlogits = row_by.get("grad.act.act.Sfooter.logits.0.y")
    if dlogits is not None:
        print(f"dlogits max_abs_diff={float(dlogits['max_abs_diff']):.6e}")
    dtoken = by_w.get("grad.weight.token_emb")
    if dtoken is not None:
        print(f"d_token_emb max_abs_diff={float(dtoken['max_abs_diff']):.6e}")
    dfn = by_w.get("grad.weight.final_ln_weight")
    if dfn is not None:
        print(f"d_final_ln_weight max_abs_diff={float(dfn['max_abs_diff']):.6e}")
    drms = row_by.get("grad.act.act.Sfooter.rmsnorm.0.output")
    if drms is not None:
        print(f"d_final_rmsnorm_out max_abs_diff={float(drms['max_abs_diff']):.6e}")
    if last_layer is not None:
        top_act = (last_layer_act[0] if last_layer_act else None)
        top_w = (last_layer_weight[0] if last_layer_weight else None)
        if top_act is not None:
            print(
                f"last_layer=L{last_layer} top_act={top_act['grad_slot']} "
                f"max_abs_diff={float(top_act['max_abs_diff']):.6e}"
            )
        if top_w is not None:
            print(
                f"last_layer=L{last_layer} top_weight={top_w['grad_weight']} "
                f"max_abs_diff={float(top_w['max_abs_diff']):.6e}"
            )
    if first_big is not None:
        print(
            "First chain slot diff > 1e-3:",
            first_big["grad_slot"],
            f"max_abs_diff={first_big['max_abs_diff']:.6e}",
        )
    if per_layer:
        worst_layer = max(
            per_layer,
            key=lambda r: float(
                max(
                    (r.get("act_max_abs_diff") or 0.0),
                    (r.get("weight_max_abs_diff") or 0.0),
                )
            ),
        )
        print(
            "Per-layer worst:",
            f"L{int(worst_layer['layer'])}",
            f"act_max={float(worst_layer.get('act_max_abs_diff') or 0.0):.6e}",
            f"weight_max={float(worst_layer.get('weight_max_abs_diff') or 0.0):.6e}",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Torch snapshot oracle for v7 CK training runtime.

Given a CK exported weight snapshot (contiguous float array over weight.* tensor slots)
and runtime metadata, reconstruct a Torch forward pass and compute same-batch loss.
This is used as a per-check-step oracle for `ck_run_v7.py --backend ck --parity-on`.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

try:
    import torch
    import torch.nn.functional as F
except Exception:  # pragma: no cover - caller handles missing torch
    torch = None  # type: ignore
    F = None  # type: ignore


def _require_torch() -> None:
    if torch is None or F is None:
        raise RuntimeError("torch is required for snapshot oracle")


def _manifest_index(manifest: Mapping[str, object]) -> Dict[str, Mapping[str, object]]:
    entries = manifest.get("entries") if isinstance(manifest, Mapping) else None
    out: Dict[str, Mapping[str, object]] = {}
    if not isinstance(entries, list):
        return out
    for row in entries:
        if not isinstance(row, Mapping):
            continue
        name = row.get("name")
        if isinstance(name, str) and name:
            out[name] = row
    return out


def _as_shape(entry: Mapping[str, object], fallback_numel: int) -> Tuple[int, ...]:
    raw = entry.get("shape")
    if isinstance(raw, list) and raw:
        shp = tuple(int(v) for v in raw)
        prod = 1
        for v in shp:
            prod *= int(v)
        if prod == int(fallback_numel):
            return shp
    return (int(fallback_numel),)


def _rms_norm(x: "torch.Tensor", gamma: "torch.Tensor", eps: float) -> "torch.Tensor":
    var = torch.mean(x * x, dim=-1, keepdim=True)
    xn = x * torch.rsqrt(var + eps)
    return xn * gamma


def _apply_affine(x: "torch.Tensor", w: "torch.Tensor", b: Optional["torch.Tensor"]) -> "torch.Tensor":
    # Handles both [in,out] and [out,in] stored layouts.
    if w.ndim != 2:
        raise ValueError(f"Expected rank-2 weight, got shape={tuple(w.shape)}")
    xin = int(x.shape[-1])
    if xin == int(w.shape[0]) and xin == int(w.shape[1]):
        # CK GEMM kernels consume B in [N,K] layout; for square matrices this is
        # ambiguous, so prefer the transposed interpretation to match runtime.
        y = torch.matmul(x, w.transpose(0, 1))
    elif xin == int(w.shape[0]):
        y = torch.matmul(x, w)
    elif xin == int(w.shape[1]):
        y = torch.matmul(x, w.transpose(0, 1))
    else:
        raise ValueError(
            f"Affine shape mismatch: x[-1]={xin}, w.shape={tuple(w.shape)}"
        )
    if b is not None:
        if b.ndim != 1:
            raise ValueError(f"Expected rank-1 bias, got shape={tuple(b.shape)}")
        if int(y.shape[-1]) != int(b.shape[0]):
            raise ValueError(
                f"Bias shape mismatch: y[-1]={int(y.shape[-1])}, b={tuple(b.shape)}"
            )
        y = y + b
    return y


def _apply_head_rms_norm(x: "torch.Tensor", gamma: Optional["torch.Tensor"], eps: float) -> "torch.Tensor":
    # x: [B, H, T, Dh]
    if gamma is None:
        return x
    var = torch.mean(x * x, dim=-1, keepdim=True)
    xn = x * torch.rsqrt(var + eps)
    return xn * gamma.view(1, 1, 1, -1)


def _apply_rope(q: "torch.Tensor", k: "torch.Tensor", theta: float) -> Tuple["torch.Tensor", "torch.Tensor"]:
    # q/k: [B,H,T,Dh]
    dh = int(q.shape[-1])
    if dh < 2:
        return q, k
    rotary_dim = dh - (dh % 2)
    if rotary_dim <= 0:
        return q, k

    device = q.device
    dtype = q.dtype
    t = int(q.shape[-2])

    half = rotary_dim // 2
    freq_seq = torch.arange(0, half, device=device, dtype=torch.float32)
    inv_freq = 1.0 / (float(theta) ** (freq_seq / max(1.0, float(half))))
    pos = torch.arange(t, device=device, dtype=torch.float32)
    freqs = torch.outer(pos, inv_freq)  # [T, half]
    cos = torch.cos(freqs).to(dtype=dtype).view(1, 1, t, half)
    sin = torch.sin(freqs).to(dtype=dtype).view(1, 1, t, half)

    def _rope_one(x: "torch.Tensor") -> "torch.Tensor":
        x_rot = x[..., :rotary_dim]
        x_pass = x[..., rotary_dim:]
        x_even = x_rot[..., 0::2]
        x_odd = x_rot[..., 1::2]
        ro_even = (x_even * cos) - (x_odd * sin)
        ro_odd = (x_even * sin) + (x_odd * cos)
        out_rot = torch.stack([ro_even, ro_odd], dim=-1).flatten(-2)
        return torch.cat([out_rot, x_pass], dim=-1)

    return _rope_one(q), _rope_one(k)


def _causal_attention(
    q: "torch.Tensor", k: "torch.Tensor", v: "torch.Tensor", *, scale: float
) -> "torch.Tensor":
    # q: [B,Hq,T,Dh], k/v: [B,Hk,T,Dh]
    hq = int(q.shape[1])
    hk = int(k.shape[1])
    if hk <= 0:
        raise ValueError("Invalid K heads")
    if hq != hk:
        if (hq % hk) != 0:
            raise ValueError(f"Unsupported GQA mapping: q_heads={hq}, kv_heads={hk}")
        rep = hq // hk
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)

    scores = torch.matmul(q, k.transpose(-2, -1)) * float(scale)
    t = int(scores.shape[-1])
    mask = torch.triu(torch.ones((t, t), dtype=torch.bool, device=scores.device), diagonal=1)
    scores = scores.masked_fill(mask.view(1, 1, t, t), float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    ctx = torch.matmul(probs, v)
    return ctx


def _decode_weight_snapshot(
    run_dir: Path,
    runtime_summary: Mapping[str, object],
    snapshot: np.ndarray,
) -> Tuple[Dict[str, "torch.Tensor"], Dict[str, int | float]]:
    _require_torch()

    manifest_path = run_dir / "weights_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing weights_manifest.json: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_idx = _manifest_index(manifest)

    cfg = manifest.get("config") if isinstance(manifest, Mapping) else None
    if not isinstance(cfg, Mapping):
        cfg = {}

    slot_rows = runtime_summary.get("tensor_slots") if isinstance(runtime_summary, Mapping) else None
    if not isinstance(slot_rows, list):
        raise ValueError("runtime_summary.tensor_slots missing")

    weight_slots = [
        row
        for row in slot_rows
        if isinstance(row, Mapping)
        and str(row.get("name", "")).startswith("weight.")
        and str(row.get("section", "")) == "weights"
    ]
    weight_slots.sort(key=lambda r: int(r.get("offset", 0)))

    expected = sum(int(r.get("numel", 0)) for r in weight_slots)
    if int(snapshot.size) < int(expected):
        raise ValueError(f"Snapshot too small: got={int(snapshot.size)} expected={int(expected)}")

    cursor = 0
    tensors: Dict[str, torch.Tensor] = {}
    for row in weight_slots:
        full = str(row.get("name"))
        name = full[len("weight."):] if full.startswith("weight.") else full
        numel = int(row.get("numel", 0))
        if numel <= 0:
            continue
        arr = snapshot[cursor : cursor + numel]
        cursor += numel

        ent = manifest_idx.get(name, {})
        shape = _as_shape(ent, numel)
        flat = np.asarray(arr, dtype=np.float32)
        vals = flat.reshape(shape)
        tensors[name] = torch.from_numpy(vals.copy())

    model_cfg = {
        "num_layers": int(cfg.get("num_layers", 0) or 0),
        "embed_dim": int(cfg.get("embed_dim", 0) or 0),
        "hidden_size": int(cfg.get("hidden_size", cfg.get("intermediate_size", 0)) or 0),
        "vocab_size": int(cfg.get("vocab_size", 0) or 0),
        "num_heads": int(cfg.get("num_heads", 0) or 0),
        "num_kv_heads": int(cfg.get("num_kv_heads", cfg.get("num_heads", 0)) or 0),
        "head_dim": int(cfg.get("head_dim", 0) or 0),
        "rope_theta": float(cfg.get("rope_theta", 10000.0) or 10000.0),
        "rms_eps": 1e-5,
    }

    # Derive missing dims from tensors when config fields are absent.
    tok = tensors.get("token_emb")
    if tok is not None and tok.ndim == 2:
        model_cfg["vocab_size"] = int(tok.shape[0])
        model_cfg["embed_dim"] = int(tok.shape[1])
    if model_cfg["head_dim"] <= 0 and model_cfg["num_heads"] > 0 and model_cfg["embed_dim"] > 0:
        model_cfg["head_dim"] = model_cfg["embed_dim"] // model_cfg["num_heads"]
    if model_cfg["num_layers"] <= 0:
        # Infer from tensor names layer.N.*
        max_layer = -1
        for n in tensors.keys():
            if n.startswith("layer."):
                parts = n.split(".")
                if len(parts) > 2 and parts[1].isdigit():
                    max_layer = max(max_layer, int(parts[1]))
        model_cfg["num_layers"] = max_layer + 1 if max_layer >= 0 else 0

    return tensors, model_cfg


class SnapshotQwenLikeOracle:
    def __init__(self, tensors: Mapping[str, "torch.Tensor"], cfg: Mapping[str, int | float]):
        _require_torch()
        self.tensors = tensors
        self.cfg = cfg

    def _get(self, name: str, *, required: bool = True) -> Optional["torch.Tensor"]:
        t = self.tensors.get(name)
        if t is None and required:
            raise KeyError(f"Missing weight tensor: {name}")
        return t

    @staticmethod
    def _npflat(x: "torch.Tensor") -> np.ndarray:
        return x.detach().cpu().float().reshape(-1).numpy().astype(np.float32, copy=False)

    @staticmethod
    def _rms_norm_with_cache(x: "torch.Tensor", gamma: "torch.Tensor", eps: float) -> tuple["torch.Tensor", "torch.Tensor"]:
        var = torch.mean(x * x, dim=-1, keepdim=True)
        rstd = torch.rsqrt(var + eps)
        xn = x * rstd
        return xn * gamma, rstd.squeeze(-1)

    def _forward_core(
        self,
        input_ids: "torch.Tensor",
        *,
        capture_slots: bool,
        saved_attn_slot_names: Optional[List[str]] = None,
    ) -> tuple["torch.Tensor", Dict[str, np.ndarray]]:
        _require_torch()
        if input_ids.ndim != 2:
            raise ValueError(f"Expected input_ids [B,T], got {tuple(input_ids.shape)}")
        bsz, tsz = int(input_ids.shape[0]), int(input_ids.shape[1])

        d_model = int(self.cfg.get("embed_dim", 0) or 0)
        n_layers = int(self.cfg.get("num_layers", 0) or 0)
        n_heads = int(self.cfg.get("num_heads", 0) or 0)
        n_kv_heads_cfg = int(self.cfg.get("num_kv_heads", n_heads) or n_heads)
        head_dim = int(self.cfg.get("head_dim", 0) or 0)
        theta = float(self.cfg.get("rope_theta", 10000.0) or 10000.0)
        eps = float(self.cfg.get("rms_eps", 1e-5) or 1e-5)

        tok = self._get("token_emb")
        assert tok is not None
        x = F.embedding(input_ids, tok)

        slot_map: Dict[str, np.ndarray] = {}
        if capture_slots:
            slot_map["act.Sheader.dense_embedding_lookup.0.out"] = self._npflat(x)

        if d_model <= 0:
            d_model = int(x.shape[-1])
        if n_heads <= 0:
            raise ValueError("Invalid num_heads in config")
        if head_dim <= 0:
            head_dim = d_model // n_heads

        saved_probs: List[np.ndarray] = []

        for layer in range(n_layers):
            pref = f"layer.{layer}."
            sp = f"act.L{layer}."

            residual = x

            ln1 = self._get(pref + "ln1_gamma")
            assert ln1 is not None
            x_norm, rstd1 = self._rms_norm_with_cache(x, ln1, eps)
            if capture_slots:
                slot_map[sp + "rmsnorm.0.output"] = self._npflat(x_norm)
                slot_map[sp + "rmsnorm.0.rstd_cache"] = self._npflat(rstd1)

            wq = self._get(pref + "wq")
            wk = self._get(pref + "wk")
            wv = self._get(pref + "wv")
            wo = self._get(pref + "wo")
            bq = self._get(pref + "bq", required=False)
            bk = self._get(pref + "bk", required=False)
            bv = self._get(pref + "bv", required=False)
            bo = self._get(pref + "bo", required=False)

            assert wq is not None and wk is not None and wv is not None and wo is not None

            q = _apply_affine(x_norm, wq, bq)
            k = _apply_affine(x_norm, wk, bk)
            v = _apply_affine(x_norm, wv, bv)

            # Runtime IR for GQA may use packed wk/wv tensors where only the
            # first (num_kv_heads * head_dim) channels are materialized.
            q_expected = max(1, int(n_heads) * int(head_dim))
            kv_expected = max(1, int(n_kv_heads_cfg) * int(head_dim))
            if int(q.shape[-1]) >= q_expected:
                q = q[..., :q_expected]
            if int(k.shape[-1]) >= kv_expected:
                k = k[..., :kv_expected]
            if int(v.shape[-1]) >= kv_expected:
                v = v[..., :kv_expected]

            if capture_slots:
                slot_map[sp + "q_proj.0.y"] = self._npflat(q)
                slot_map[sp + "k_proj.0.y"] = self._npflat(k)
                slot_map[sp + "v_proj.0.y"] = self._npflat(v)

            q_out = int(q.shape[-1])
            k_out = int(k.shape[-1])
            q_heads = n_heads if (n_heads * head_dim) == q_out else max(1, q_out // max(1, head_dim))
            kv_heads_nom = n_kv_heads_cfg if (n_kv_heads_cfg * head_dim) == k_out else max(1, k_out // max(1, head_dim))

            qh = q.view(bsz, tsz, q_heads, head_dim).permute(0, 2, 1, 3).contiguous()
            kh = k.view(bsz, tsz, kv_heads_nom, head_dim).permute(0, 2, 1, 3).contiguous()
            vh = v.view(bsz, tsz, kv_heads_nom, head_dim).permute(0, 2, 1, 3).contiguous()

            qn = self._get(pref + "q_norm", required=False)
            kn = self._get(pref + "k_norm", required=False)
            qh = _apply_head_rms_norm(qh, qn, eps)
            kh = _apply_head_rms_norm(kh, kn, eps)
            if capture_slots:
                qn_flat = qh.permute(0, 2, 1, 3).contiguous().view(bsz, tsz, -1)
                kn_flat = kh.permute(0, 2, 1, 3).contiguous().view(bsz, tsz, -1)
                slot_map[sp + "qk_norm.0.q"] = self._npflat(qn_flat)
                slot_map[sp + "qk_norm.0.k"] = self._npflat(kn_flat)

            qh, kh = _apply_rope(qh, kh, theta)
            if capture_slots:
                qr_flat = qh.permute(0, 2, 1, 3).contiguous().view(bsz, tsz, -1)
                kr_flat = kh.permute(0, 2, 1, 3).contiguous().view(bsz, tsz, -1)
                slot_map[sp + "rope_qk.0.q"] = self._npflat(qr_flat)
                slot_map[sp + "rope_qk.0.k"] = self._npflat(kr_flat)

            scale = 1.0 / math.sqrt(float(max(1, head_dim)))
            hq = int(qh.shape[1])
            hk = int(kh.shape[1])
            if hk <= 0:
                raise ValueError("Invalid K heads")
            kh_eff = kh
            vh_eff = vh
            if hq != hk:
                if (hq % hk) != 0:
                    raise ValueError(f"Unsupported GQA mapping: q_heads={hq}, kv_heads={hk}")
                rep = hq // hk
                kh_eff = kh.repeat_interleave(rep, dim=1)
                vh_eff = vh.repeat_interleave(rep, dim=1)

            scores = torch.matmul(qh, kh_eff.transpose(-2, -1)) * float(scale)
            t = int(scores.shape[-1])
            mask = torch.triu(torch.ones((t, t), dtype=torch.bool, device=scores.device), diagonal=1)
            scores = scores.masked_fill(mask.view(1, 1, t, t), float("-inf"))
            probs = torch.softmax(scores, dim=-1)
            ctx = torch.matmul(probs, vh_eff)
            if capture_slots:
                saved_probs.append(self._npflat(probs))

            ctx = ctx.permute(0, 2, 1, 3).contiguous().view(bsz, tsz, -1)
            if capture_slots:
                slot_map[sp + "attn.0.out"] = self._npflat(ctx)

            attn_out = _apply_affine(ctx, wo, bo)
            if capture_slots:
                slot_map[sp + "out_proj.0.y"] = self._npflat(attn_out)
            x = residual + attn_out
            if capture_slots:
                slot_map[sp + "residual_add.0.out"] = self._npflat(x)

            residual2 = x
            ln2 = self._get(pref + "ln2_gamma")
            assert ln2 is not None
            x_norm2, rstd2 = self._rms_norm_with_cache(x, ln2, eps)
            if capture_slots:
                slot_map[sp + "rmsnorm.1.output"] = self._npflat(x_norm2)
                slot_map[sp + "rmsnorm.1.rstd_cache"] = self._npflat(rstd2)

            w1 = self._get(pref + "w1")
            w2 = self._get(pref + "w2")
            b1 = self._get(pref + "b1", required=False)
            b2 = self._get(pref + "b2", required=False)
            assert w1 is not None and w2 is not None

            gate_up = _apply_affine(x_norm2, w1, b1)
            if capture_slots:
                slot_map[sp + "mlp_gate_up.0.y"] = self._npflat(gate_up)
            if int(gate_up.shape[-1]) % 2 != 0:
                raise ValueError(f"Invalid SwiGLU width at {pref}w1: {tuple(gate_up.shape)}")
            half = int(gate_up.shape[-1]) // 2
            gate = gate_up[..., :half]
            up = gate_up[..., half:]
            mlp_hidden = F.silu(gate) * up
            if capture_slots:
                slot_map[sp + "silu_mul.0.out"] = self._npflat(mlp_hidden)
            mlp_out = _apply_affine(mlp_hidden, w2, b2)
            if capture_slots:
                slot_map[sp + "mlp_down.0.y"] = self._npflat(mlp_out)
            x = residual2 + mlp_out
            if capture_slots:
                slot_map[sp + "residual_add.1.out"] = self._npflat(x)

        final_ln = self._get("final_ln_weight")
        assert final_ln is not None
        x, rstd_final = self._rms_norm_with_cache(x, final_ln, eps)
        if capture_slots:
            slot_map["act.Sfooter.rmsnorm.0.output"] = self._npflat(x)
            slot_map["act.Sfooter.rmsnorm.0.rstd_cache"] = self._npflat(rstd_final)

        lm_head = self._get("output.weight", required=False)
        if lm_head is None:
            lm_head = tok
        logits = torch.matmul(x, lm_head.transpose(0, 1))
        if capture_slots:
            slot_map["act.Sfooter.logits.0.y"] = self._npflat(logits)

            if saved_attn_slot_names:
                for idx, name in enumerate(saved_attn_slot_names):
                    if idx < len(saved_probs):
                        slot_map[name] = saved_probs[idx]

        return logits, slot_map

    def forward(self, input_ids: "torch.Tensor") -> "torch.Tensor":
        logits, _ = self._forward_core(input_ids, capture_slots=False, saved_attn_slot_names=None)
        return logits

    def forward_with_slots(
        self,
        input_ids: "torch.Tensor",
        *,
        saved_attn_slot_names: Optional[List[str]] = None,
    ) -> tuple["torch.Tensor", Dict[str, np.ndarray]]:
        return self._forward_core(input_ids, capture_slots=True, saved_attn_slot_names=saved_attn_slot_names)


def _saved_attn_slot_names(runtime_summary: Mapping[str, object]) -> List[str]:
    rows = runtime_summary.get("tensor_slots") if isinstance(runtime_summary, Mapping) else None
    if not isinstance(rows, list):
        return []

    parsed: List[tuple[int, str]] = []
    fallback: List[str] = []
    import re
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        if str(row.get("section", "")) != "saved":
            continue
        name = str(row.get("name", ""))
        if "attn_weights" not in name:
            continue
        m = re.search(r"op(\d+)", name)
        if m:
            parsed.append((int(m.group(1)), name))
        else:
            fallback.append(name)

    if parsed:
        parsed.sort(key=lambda x: x[0])
        return [n for _, n in parsed]
    fallback.sort()
    return fallback


def compute_loss_logits_and_slots_from_snapshot_array(
    run_dir: Path,
    runtime_summary: Mapping[str, object],
    snapshot: np.ndarray,
    input_ids: Sequence[int],
    targets: Sequence[int],
) -> tuple[float, np.ndarray, Dict[str, np.ndarray]]:
    _require_torch()

    if not isinstance(snapshot, np.ndarray):
        snapshot = np.asarray(snapshot, dtype=np.float32)
    if snapshot.dtype != np.float32:
        snapshot = snapshot.astype(np.float32, copy=False)

    weights, cfg = _decode_weight_snapshot(run_dir, runtime_summary, snapshot)
    model = SnapshotQwenLikeOracle(weights, cfg)

    x = torch.tensor(list(input_ids), dtype=torch.long).view(1, -1)
    y = torch.tensor(list(targets), dtype=torch.long).view(1, -1)

    saved_names = _saved_attn_slot_names(runtime_summary)
    logits, slot_map = model.forward_with_slots(x, saved_attn_slot_names=saved_names)
    if int(logits.shape[1]) != int(y.shape[1]):
        raise ValueError(
            f"Oracle logits/targets shape mismatch: logits={tuple(logits.shape)} targets={tuple(y.shape)}"
        )

    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1), reduction="mean")
    loss_val = float(loss.detach().cpu().item())
    logits_flat = logits.detach().cpu().float().reshape(-1).numpy().astype(np.float32, copy=False)
    slot_map["aux.loss"] = np.array([loss_val], dtype=np.float32)
    return loss_val, logits_flat, slot_map


def compute_loss_and_logits_from_snapshot_array(
    run_dir: Path,
    runtime_summary: Mapping[str, object],
    snapshot: np.ndarray,
    input_ids: Sequence[int],
    targets: Sequence[int],
) -> tuple[float, np.ndarray]:
    loss, logits, _ = compute_loss_logits_and_slots_from_snapshot_array(
        run_dir=run_dir,
        runtime_summary=runtime_summary,
        snapshot=snapshot,
        input_ids=input_ids,
        targets=targets,
    )
    return float(loss), logits


def compute_loss_from_snapshot_array(
    run_dir: Path,
    runtime_summary: Mapping[str, object],
    snapshot: np.ndarray,
    input_ids: Sequence[int],
    targets: Sequence[int],
) -> float:
    loss, _ = compute_loss_and_logits_from_snapshot_array(
        run_dir=run_dir,
        runtime_summary=runtime_summary,
        snapshot=snapshot,
        input_ids=input_ids,
        targets=targets,
    )
    return float(loss)

def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Compute Torch oracle loss from CK weight snapshot")
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--runtime-summary", type=Path, required=True)
    ap.add_argument("--snapshot", type=Path, required=True)
    ap.add_argument("--input-ids", type=str, required=True, help="Comma-separated token ids")
    ap.add_argument("--targets", type=str, required=True, help="Comma-separated token ids")
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args()

    if torch is None:
        raise RuntimeError("torch is not installed")

    rs = json.loads(args.runtime_summary.read_text(encoding="utf-8"))
    snap = np.frombuffer(args.snapshot.read_bytes(), dtype=np.float32)
    x = [int(v.strip()) for v in args.input_ids.split(",") if v.strip()]
    y = [int(v.strip()) for v in args.targets.split(",") if v.strip()]

    loss, logits = compute_loss_and_logits_from_snapshot_array(args.run_dir, rs, snap, x, y)
    out = {"loss": float(loss), "logits_numel": int(logits.size)}
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    else:
        print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

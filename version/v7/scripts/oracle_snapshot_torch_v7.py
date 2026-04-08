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
import build_ir_v7

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


def _first_present(*values):
    for value in values:
        if value is not None:
            return value
    return None


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


def _normalize_mlp_activation(name: object) -> str:
    text = str(name or "").strip().lower()
    if text in {"geglu", "gelu_glu", "gelu"}:
        return "geglu"
    return "swiglu"


def _template_mlp_activation(manifest: Mapping[str, object]) -> str:
    template = manifest.get("template") if isinstance(manifest, Mapping) else None
    template = template if isinstance(template, Mapping) else {}
    contract = template.get("contract") if isinstance(template, Mapping) else None
    contract = contract if isinstance(contract, Mapping) else {}
    block_contract = contract.get("block_contract") if isinstance(contract, Mapping) else None
    block_contract = block_contract if isinstance(block_contract, Mapping) else {}
    flags = template.get("flags") if isinstance(template, Mapping) else None
    flags = flags if isinstance(flags, Mapping) else {}
    activation = (
        block_contract.get("activation")
        or flags.get("activation")
        or "swiglu"
    )
    return _normalize_mlp_activation(activation)


def _activation_slot_name(activation: object) -> str:
    return "geglu" if _normalize_mlp_activation(activation) == "geglu" else "silu_mul"


def _normalize_attention_variant(name: object) -> str:
    text = str(name or "").strip().lower()
    if text in {"sliding", "sliding_window", "attn_sliding"}:
        return "sliding_window"
    if text in {"hybrid_recurrent_attention", "hybrid", "qwen35_hybrid"}:
        return "hybrid_recurrent_attention"
    return "dense"


def _template_attention_variant(manifest: Mapping[str, object]) -> str:
    template = manifest.get("template") if isinstance(manifest, Mapping) else None
    template = template if isinstance(template, Mapping) else {}
    contract = template.get("contract") if isinstance(template, Mapping) else None
    contract = contract if isinstance(contract, Mapping) else {}
    attention_contract = contract.get("attention_contract") if isinstance(contract, Mapping) else None
    attention_contract = attention_contract if isinstance(attention_contract, Mapping) else {}
    flags = template.get("flags") if isinstance(template, Mapping) else None
    flags = flags if isinstance(flags, Mapping) else {}
    variant = (
        attention_contract.get("attn_variant")
        or flags.get("attention")
        or "dense"
    )
    return _normalize_attention_variant(variant)


def _attention_slot_name(variant: object) -> str:
    return "attn_sliding" if _normalize_attention_variant(variant) == "sliding_window" else "attn"


def _layer_kind(cfg: Mapping[str, object], layer_idx: int) -> str:
    raw = cfg.get("layer_kinds")
    if isinstance(raw, list) and layer_idx < len(raw):
        return str(raw[layer_idx] or "").strip().lower() or "recurrent"
    interval = int(cfg.get("full_attention_interval", 0) or 0)
    if interval > 0 and ((layer_idx + 1) % interval) == 0:
        return "full_attention"
    return "recurrent" if _normalize_attention_variant(cfg.get("attn_variant")) == "hybrid_recurrent_attention" else "dense"


def _recurrent_conv_state_update(
    q: "torch.Tensor",
    k: "torch.Tensor",
    v: "torch.Tensor",
    *,
    history_len: int,
    state_in: Optional["torch.Tensor"] = None,
) -> tuple["torch.Tensor", "torch.Tensor"]:
    # Runtime contract:
    #   state_in:  [B, C, history]
    #   conv_x:    [B, C, history + T]
    #   state_out: [B, C, history]
    packed = torch.cat([q, k, v], dim=-1).transpose(1, 2).contiguous()
    bsz, channels = int(packed.shape[0]), int(packed.shape[1])
    history = max(0, int(history_len))
    if state_in is None:
        state_in = torch.zeros((bsz, channels, history), dtype=packed.dtype, device=packed.device)
    conv_x = torch.cat([state_in, packed], dim=-1)
    if history > 0:
        state_out = conv_x[:, :, -history:].contiguous()
    else:
        state_out = torch.zeros((bsz, channels, 0), dtype=packed.dtype, device=packed.device)
    return conv_x.contiguous(), state_out


def _depthwise_causal_conv(conv_x: "torch.Tensor", kernel: "torch.Tensor") -> "torch.Tensor":
    # conv_x: [B,C,K-1+T], kernel: [C,K] -> out: [B,T,C]
    bsz, channels, width = int(conv_x.shape[0]), int(conv_x.shape[1]), int(conv_x.shape[2])
    kernel_size = int(kernel.shape[1])
    tsz = max(0, width - kernel_size + 1)
    out = torch.zeros((bsz, tsz, channels), dtype=conv_x.dtype, device=conv_x.device)
    for tap in range(kernel_size):
        window = conv_x[:, :, tap : tap + tsz].transpose(1, 2).contiguous()
        out = out + window * kernel[:, tap].view(1, 1, channels)
    return out


def _recurrent_qk_l2_norm(
    q: "torch.Tensor",
    k: "torch.Tensor",
    *,
    num_heads: int,
    head_dim: int,
    eps: float,
) -> tuple["torch.Tensor", "torch.Tensor"]:
    bsz, tsz = int(q.shape[0]), int(q.shape[1])
    qh = q.view(bsz, tsz, num_heads, head_dim)
    kh = k.view(bsz, tsz, num_heads, head_dim)
    qh = qh / torch.sqrt(torch.sum(qh * qh, dim=-1, keepdim=True) + float(eps))
    kh = kh / torch.sqrt(torch.sum(kh * kh, dim=-1, keepdim=True) + float(eps))
    return qh.view(bsz, tsz, -1), kh.view(bsz, tsz, -1)


def _torch_deltanet_sequence(
    q: "torch.Tensor",
    k: "torch.Tensor",
    v: "torch.Tensor",
    g: "torch.Tensor",
    beta: "torch.Tensor",
    *,
    num_heads: int,
    state_dim: int,
) -> tuple["torch.Tensor", "torch.Tensor"]:
    # q/k/v: [B,T,H*D], g/beta: [B,T,H]
    bsz, tsz = int(q.shape[0]), int(q.shape[1])
    qh = q.view(bsz, tsz, num_heads, state_dim)
    kh = k.view(bsz, tsz, num_heads, state_dim)
    vh = v.view(bsz, tsz, num_heads, state_dim)
    state = torch.zeros((bsz, num_heads, state_dim, state_dim), dtype=q.dtype, device=q.device)
    outs: List["torch.Tensor"] = []
    for tok in range(tsz):
        q_t = qh[:, tok, :, :]
        k_t = kh[:, tok, :, :]
        v_t = vh[:, tok, :, :]
        g_t = g[:, tok, :]
        beta_t = beta[:, tok, :]
        q_scale = 1.0 / math.sqrt(float(max(1, state_dim)))
        q_hat = q_t * q_scale
        beta_s = torch.sigmoid(beta_t).unsqueeze(-1)
        gate = torch.exp(g_t).unsqueeze(-1).unsqueeze(-1)
        gated_state = state * gate
        kv_mem = torch.matmul(gated_state.transpose(-1, -2), k_t.unsqueeze(-1)).squeeze(-1)
        delta = (v_t - kv_mem) * beta_s
        state = gated_state + torch.matmul(k_t.unsqueeze(-1), delta.unsqueeze(-2))
        out_t = torch.matmul(state.transpose(-1, -2), q_hat.unsqueeze(-1)).squeeze(-1)
        outs.append(out_t.reshape(bsz, -1))
    out = torch.stack(outs, dim=1) if outs else torch.zeros((bsz, 0, num_heads * state_dim), dtype=q.dtype, device=q.device)
    return out, state


def _recurrent_norm_gate(
    x: "torch.Tensor",
    gate: "torch.Tensor",
    weight: "torch.Tensor",
    *,
    num_heads: int,
    head_dim: int,
    eps: float,
) -> "torch.Tensor":
    bsz, tsz = int(x.shape[0]), int(x.shape[1])
    xh = x.view(bsz, tsz, num_heads, head_dim)
    gh = gate.view(bsz, tsz, num_heads, head_dim)
    var = torch.mean(xh * xh, dim=-1, keepdim=True)
    xn = xh * torch.rsqrt(var + eps)
    return (xn * weight.view(1, 1, 1, head_dim) * F.silu(gh)).view(bsz, tsz, num_heads * head_dim)


def _normalize_rope_layout(name: object) -> str:
    text = str(name or "").strip().lower()
    if text in {"pairwise", "interleaved", "llama_pairwise"}:
        return "pairwise"
    return "split"


def _template_rope_layout(manifest: Mapping[str, object]) -> str:
    template = manifest.get("template") if isinstance(manifest, Mapping) else None
    template = template if isinstance(template, Mapping) else {}
    contract = template.get("contract") if isinstance(template, Mapping) else None
    contract = contract if isinstance(contract, Mapping) else {}
    attention_contract = contract.get("attention_contract") if isinstance(contract, Mapping) else None
    attention_contract = attention_contract if isinstance(attention_contract, Mapping) else {}
    cfg = manifest.get("config") if isinstance(manifest, Mapping) else None
    cfg = cfg if isinstance(cfg, Mapping) else {}
    layout = (
        attention_contract.get("rope_layout")
        or cfg.get("rope_layout")
        or "split"
    )
    return _normalize_rope_layout(layout)


def _apply_glu_activation(gate_up: "torch.Tensor", activation: object) -> "torch.Tensor":
    act = _normalize_mlp_activation(activation)
    if int(gate_up.shape[-1]) % 2 != 0:
        raise ValueError(f"Invalid {act} width: {tuple(gate_up.shape)}")
    half = int(gate_up.shape[-1]) // 2
    gate = gate_up[..., :half]
    up = gate_up[..., half:]
    if act == "geglu":
        return F.gelu(gate, approximate="tanh") * up
    return F.silu(gate) * up


def _apply_head_rms_norm(x: "torch.Tensor", gamma: Optional["torch.Tensor"], eps: float) -> "torch.Tensor":
    # x: [B, H, T, Dh]
    if gamma is None:
        return x
    var = torch.mean(x * x, dim=-1, keepdim=True)
    xn = x * torch.rsqrt(var + eps)
    return xn * gamma.view(1, 1, 1, -1)


def _apply_rope(
    q: "torch.Tensor",
    k: "torch.Tensor",
    theta: float,
    rope_layout: object = "split",
) -> Tuple["torch.Tensor", "torch.Tensor"]:
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
    layout = _normalize_rope_layout(rope_layout)

    def _rope_one(x: "torch.Tensor") -> "torch.Tensor":
        x_rot = x[..., :rotary_dim]
        x_pass = x[..., rotary_dim:]
        if layout == "pairwise":
            x0 = x_rot[..., 0:rotary_dim:2]
            x1 = x_rot[..., 1:rotary_dim:2]
        else:
            x0 = x_rot[..., :half]
            x1 = x_rot[..., half:rotary_dim]
        r0 = (x0 * cos) - (x1 * sin)
        r1 = (x0 * sin) + (x1 * cos)
        if layout == "pairwise":
            out_rot = torch.empty_like(x_rot)
            out_rot[..., 0:rotary_dim:2] = r0
            out_rot[..., 1:rotary_dim:2] = r1
        else:
            out_rot = torch.cat([r0, r1], dim=-1)
        return torch.cat([out_rot, x_pass], dim=-1)

    return _rope_one(q), _rope_one(k)


def _causal_attention(
    q: "torch.Tensor",
    k: "torch.Tensor",
    v: "torch.Tensor",
    *,
    scale: float,
    sliding_window: int = 0,
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
    q_idx = torch.arange(t, dtype=torch.int64, device=scores.device).view(t, 1)
    k_idx = torch.arange(t, dtype=torch.int64, device=scores.device).view(1, t)
    mask = k_idx > q_idx
    if int(sliding_window) > 0:
        mask = torch.logical_or(mask, k_idx < (q_idx - int(sliding_window) + 1))
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

    model_cfg = build_ir_v7._normalize_manifest_config(dict(cfg))
    model_cfg["rope_layout"] = _template_rope_layout(manifest)
    model_cfg["rms_eps"] = float(model_cfg.get("rms_eps", 1e-5) or 1e-5)
    model_cfg["mlp_activation"] = _template_mlp_activation(manifest)
    model_cfg["attn_variant"] = _template_attention_variant(manifest)
    model_cfg["sliding_window"] = int(model_cfg.get("sliding_window", manifest.get("sliding_window", 0)) or 0)

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
        rope_layout = _normalize_rope_layout(self.cfg.get("rope_layout", "split"))
        eps = float(self.cfg.get("rms_eps", 1e-5) or 1e-5)
        mlp_activation = _normalize_mlp_activation(self.cfg.get("mlp_activation", "swiglu"))
        mlp_activation_slot = _activation_slot_name(mlp_activation)
        attn_variant = _normalize_attention_variant(self.cfg.get("attn_variant", "dense"))
        attn_slot = _attention_slot_name(attn_variant)
        sliding_window = int(self.cfg.get("sliding_window", 0) or 0)

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
            layer_kind = _layer_kind(self.cfg, layer)

            residual = x

            ln1 = _first_present(
                self._get(pref + "attn_norm", required=False),
                self._get(pref + "ln1_gamma", required=False),
            )
            assert ln1 is not None
            x_norm, rstd1 = self._rms_norm_with_cache(x, ln1, eps)
            if capture_slots:
                slot_map[sp + "rmsnorm.0.output"] = self._npflat(x_norm)
                slot_map[sp + "rmsnorm.0.rstd_cache"] = self._npflat(rstd1)

            if layer_kind == "recurrent":
                recurrent_q = int(self.cfg.get("q_dim", 0) or 0)
                recurrent_k = int(self.cfg.get("k_dim", recurrent_q) or recurrent_q)
                recurrent_v = int(self.cfg.get("v_dim", 0) or 0)
                gate_dim = int(self.cfg.get("gate_dim", self.cfg.get("ssm_time_step_rank", n_heads)) or n_heads)
                recurrent_head_dim = int(self.cfg.get("recurrent_head_dim", max(1, recurrent_v // max(1, gate_dim))) or max(1, recurrent_v // max(1, gate_dim)))

                packed_qkv = _apply_affine(x_norm, self._get(pref + "attn_qkv"), None)
                z = _apply_affine(x_norm, self._get(pref + "attn_gate"), None)
                alpha = _apply_affine(x_norm, self._get(pref + "ssm_alpha"), None)
                beta = _apply_affine(x_norm, self._get(pref + "ssm_beta"), None)
                if capture_slots:
                    slot_map[sp + "recurrent_qkv_proj.0.y"] = self._npflat(packed_qkv)
                    slot_map[sp + "recurrent_gate_proj.0.y"] = self._npflat(z)
                    slot_map[sp + "recurrent_alpha_proj.0.y"] = self._npflat(alpha)
                    slot_map[sp + "recurrent_beta_proj.0.y"] = self._npflat(beta)

                q_pre = packed_qkv[..., :recurrent_q]
                k_pre = packed_qkv[..., recurrent_q : recurrent_q + recurrent_k]
                v_pre = packed_qkv[..., recurrent_q + recurrent_k : recurrent_q + recurrent_k + recurrent_v]
                if capture_slots:
                    slot_map[sp + "recurrent_split_qkv.0.q"] = self._npflat(q_pre)
                    slot_map[sp + "recurrent_split_qkv.0.k"] = self._npflat(k_pre)
                    slot_map[sp + "recurrent_split_qkv.0.v"] = self._npflat(v_pre)

                dt_bias = self._get(pref + "ssm_dt_bias")
                ssm_a = self._get(pref + "ssm_a")
                gate = F.softplus(alpha + dt_bias.view(1, 1, -1)) * ssm_a.view(1, 1, -1)
                if capture_slots:
                    slot_map[sp + "recurrent_dt_gate.0.gate"] = self._npflat(gate)

                history_len = int(self.cfg.get("ssm_conv_history", max(0, int(self.cfg.get("ssm_conv_kernel", 0) or 0) - 1)) or 0)
                conv_x, conv_state_out = _recurrent_conv_state_update(
                    q_pre,
                    k_pre,
                    v_pre,
                    history_len=history_len,
                )
                if capture_slots:
                    slot_map[sp + "recurrent_conv_state_update.0.conv_x"] = self._npflat(conv_x)
                    slot_map[sp + "recurrent_conv_state_update.0.state_out"] = self._npflat(conv_state_out)

                conv_raw = _depthwise_causal_conv(conv_x, self._get(pref + "ssm_conv1d"))
                if capture_slots:
                    slot_map[sp + "recurrent_ssm_conv.0.out"] = self._npflat(conv_raw)
                conv_act = F.silu(conv_raw)
                if capture_slots:
                    slot_map[sp + "recurrent_silu.0.out"] = self._npflat(conv_act)

                q = conv_act[..., :recurrent_q]
                k = conv_act[..., recurrent_q : recurrent_q + recurrent_k]
                v = conv_act[..., recurrent_q + recurrent_k : recurrent_q + recurrent_k + recurrent_v]
                if capture_slots:
                    slot_map[sp + "recurrent_split_conv_qkv.0.q"] = self._npflat(q)
                    slot_map[sp + "recurrent_split_conv_qkv.0.k"] = self._npflat(k)
                    slot_map[sp + "recurrent_split_conv_qkv.0.v"] = self._npflat(v)

                q, k = _recurrent_qk_l2_norm(q, k, num_heads=gate_dim, head_dim=recurrent_head_dim, eps=eps)
                if capture_slots:
                    slot_map[sp + "recurrent_qk_l2_norm.0.q"] = self._npflat(q)
                    slot_map[sp + "recurrent_qk_l2_norm.0.k"] = self._npflat(k)

                rec_out, _ = _torch_deltanet_sequence(
                    q,
                    k,
                    v,
                    gate,
                    beta,
                    num_heads=gate_dim,
                    state_dim=recurrent_head_dim,
                )
                if capture_slots:
                    slot_map[sp + "recurrent_core.0.out"] = self._npflat(rec_out)

                normed = _recurrent_norm_gate(
                    rec_out,
                    z,
                    self._get(pref + "ssm_norm"),
                    num_heads=gate_dim,
                    head_dim=recurrent_head_dim,
                    eps=eps,
                )
                if capture_slots:
                    slot_map[sp + "recurrent_norm_gate.0.out"] = self._npflat(normed)

                attn_out = _apply_affine(normed, self._get(pref + "ssm_out"), None)
                if capture_slots:
                    slot_map[sp + "recurrent_out_proj.0.y"] = self._npflat(attn_out)
                x = residual + attn_out
                if capture_slots:
                    slot_map[sp + "residual_add.0.out"] = self._npflat(x)
            else:
                wq = _first_present(
                    self._get(pref + "attn_q_gate", required=False),
                    self._get(pref + "wq", required=False),
                )
                wk = _first_present(
                    self._get(pref + "attn_k", required=False),
                    self._get(pref + "wk", required=False),
                )
                wv = _first_present(
                    self._get(pref + "attn_v", required=False),
                    self._get(pref + "wv", required=False),
                )
                wo = _first_present(
                    self._get(pref + "attn_output", required=False),
                    self._get(pref + "wo", required=False),
                )
                bq = self._get(pref + "bq", required=False)
                bk = self._get(pref + "bk", required=False)
                bv = self._get(pref + "bv", required=False)
                bo = self._get(pref + "bo", required=False)

                assert wq is not None and wk is not None and wv is not None and wo is not None

                if layer_kind == "full_attention" and self._get(pref + "attn_q_gate", required=False) is not None:
                    packed_qg = _apply_affine(x_norm, wq, bq)
                    attn_out_dim = int(self.cfg.get("attn_out_dim", n_heads * head_dim) or (n_heads * head_dim))
                    attn_gate_dim = int(self.cfg.get("attn_gate_dim", attn_out_dim) or attn_out_dim)
                    q = packed_qg[..., :attn_out_dim]
                    gate = packed_qg[..., attn_out_dim : attn_out_dim + attn_gate_dim]
                    if capture_slots:
                        slot_map[sp + "q_gate_proj.0.y"] = self._npflat(packed_qg)
                        slot_map[sp + "split_q_gate.0.q"] = self._npflat(q)
                        slot_map[sp + "split_q_gate.0.gate"] = self._npflat(gate)
                else:
                    q = _apply_affine(x_norm, wq, bq)
                    gate = None
                    if capture_slots:
                        slot_map[sp + "q_proj.0.y"] = self._npflat(q)

                k = _apply_affine(x_norm, wk, bk)
                v = _apply_affine(x_norm, wv, bv)

                q_expected = max(1, int(n_heads) * int(head_dim))
                kv_expected = max(1, int(n_kv_heads_cfg) * int(head_dim))
                if int(q.shape[-1]) >= q_expected:
                    q = q[..., :q_expected]
                if int(k.shape[-1]) >= kv_expected:
                    k = k[..., :kv_expected]
                if int(v.shape[-1]) >= kv_expected:
                    v = v[..., :kv_expected]

                if capture_slots:
                    slot_map[sp + "k_proj.0.y"] = self._npflat(k)
                    slot_map[sp + "v_proj.0.y"] = self._npflat(v)

                q_out = int(q.shape[-1])
                k_out = int(k.shape[-1])
                q_heads = n_heads if (n_heads * head_dim) == q_out else max(1, q_out // max(1, head_dim))
                kv_heads_nom = n_kv_heads_cfg if (n_kv_heads_cfg * head_dim) == k_out else max(1, k_out // max(1, head_dim))

                qh = q.view(bsz, tsz, q_heads, head_dim).permute(0, 2, 1, 3).contiguous()
                kh = k.view(bsz, tsz, kv_heads_nom, head_dim).permute(0, 2, 1, 3).contiguous()
                vh = v.view(bsz, tsz, kv_heads_nom, head_dim).permute(0, 2, 1, 3).contiguous()

                qn = _first_present(
                    self._get(pref + "attn_q_norm", required=False),
                    self._get(pref + "q_norm", required=False),
                )
                kn = _first_present(
                    self._get(pref + "attn_k_norm", required=False),
                    self._get(pref + "k_norm", required=False),
                )
                qh = _apply_head_rms_norm(qh, qn, eps)
                kh = _apply_head_rms_norm(kh, kn, eps)
                if capture_slots:
                    qn_flat = qh.permute(0, 2, 1, 3).contiguous().view(bsz, tsz, -1)
                    kn_flat = kh.permute(0, 2, 1, 3).contiguous().view(bsz, tsz, -1)
                    slot_map[sp + "qk_norm.0.q"] = self._npflat(qn_flat)
                    slot_map[sp + "qk_norm.0.k"] = self._npflat(kn_flat)

                qh, kh = _apply_rope(qh, kh, theta, rope_layout)
                if capture_slots:
                    qr_flat = qh.permute(0, 2, 1, 3).contiguous().view(bsz, tsz, -1)
                    kr_flat = kh.permute(0, 2, 1, 3).contiguous().view(bsz, tsz, -1)
                    slot_map[sp + "rope_qk.0.q"] = self._npflat(qr_flat)
                    slot_map[sp + "rope_qk.0.k"] = self._npflat(kr_flat)

                scale = 1.0 / math.sqrt(float(max(1, head_dim)))
                ctx = _causal_attention(
                    qh,
                    kh,
                    vh,
                    scale=scale,
                    sliding_window=(sliding_window if attn_variant == "sliding_window" else 0),
                )
                if capture_slots and attn_variant != "sliding_window":
                    scores = torch.matmul(
                        qh,
                        (kh.repeat_interleave(int(qh.shape[1] // max(1, kh.shape[1])), dim=1) if int(qh.shape[1]) != int(kh.shape[1]) else kh).transpose(-2, -1),
                    ) * float(scale)
                    t = int(scores.shape[-1])
                    q_idx = torch.arange(t, dtype=torch.int64, device=scores.device).view(t, 1)
                    k_idx = torch.arange(t, dtype=torch.int64, device=scores.device).view(1, t)
                    scores = scores.masked_fill((k_idx > q_idx).view(1, 1, t, t), float("-inf"))
                    saved_probs.append(self._npflat(torch.softmax(scores, dim=-1)))

                ctx = ctx.permute(0, 2, 1, 3).contiguous().view(bsz, tsz, -1)
                if capture_slots:
                    slot_map[sp + f"{attn_slot}.0.out"] = self._npflat(ctx)
                if gate is not None:
                    ctx = ctx * torch.sigmoid(gate)
                    if capture_slots:
                        slot_map[sp + "attn_gate_sigmoid_mul.0.out"] = self._npflat(ctx)

                attn_out = _apply_affine(ctx, wo, bo)
                if capture_slots:
                    slot_map[sp + "out_proj.0.y"] = self._npflat(attn_out)
                x = residual + attn_out
                if capture_slots:
                    slot_map[sp + "residual_add.0.out"] = self._npflat(x)

            residual2 = x
            ln2 = _first_present(
                self._get(pref + "post_attention_norm", required=False),
                self._get(pref + "ln2_gamma", required=False),
            )
            assert ln2 is not None
            x_norm2, rstd2 = self._rms_norm_with_cache(x, ln2, eps)
            if capture_slots:
                slot_map[sp + "rmsnorm.1.output"] = self._npflat(x_norm2)
                slot_map[sp + "rmsnorm.1.rstd_cache"] = self._npflat(rstd2)

            w1 = _first_present(
                self._get(pref + "ffn_gate", required=False),
                self._get(pref + "w1", required=False),
            )
            w2 = _first_present(
                self._get(pref + "ffn_down", required=False),
                self._get(pref + "w2", required=False),
            )
            b1 = self._get(pref + "b1", required=False)
            b2 = self._get(pref + "b2", required=False)
            assert w1 is not None and w2 is not None

            gate_up = _apply_affine(x_norm2, w1, b1)
            if capture_slots:
                slot_map[sp + "mlp_gate_up.0.y"] = self._npflat(gate_up)
            mlp_hidden = _apply_glu_activation(gate_up, mlp_activation)
            if capture_slots:
                slot_map[sp + f"{mlp_activation_slot}.0.out"] = self._npflat(mlp_hidden)
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
    valid_tokens: Optional[int] = None,
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

    active = int(valid_tokens) if valid_tokens is not None else int(y.shape[1])
    active = max(1, min(active, int(logits.shape[1]), int(y.shape[1])))
    loss = F.cross_entropy(
        logits[:, :active, :].reshape(-1, logits.shape[-1]),
        y[:, :active].reshape(-1),
        reduction="mean",
    )
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
    valid_tokens: Optional[int] = None,
) -> tuple[float, np.ndarray]:
    loss, logits, _ = compute_loss_logits_and_slots_from_snapshot_array(
        run_dir=run_dir,
        runtime_summary=runtime_summary,
        snapshot=snapshot,
        input_ids=input_ids,
        targets=targets,
        valid_tokens=valid_tokens,
    )
    return float(loss), logits


def compute_loss_from_snapshot_array(
    run_dir: Path,
    runtime_summary: Mapping[str, object],
    snapshot: np.ndarray,
    input_ids: Sequence[int],
    targets: Sequence[int],
    valid_tokens: Optional[int] = None,
) -> float:
    loss, _ = compute_loss_and_logits_from_snapshot_array(
        run_dir=run_dir,
        runtime_summary=runtime_summary,
        snapshot=snapshot,
        input_ids=input_ids,
        targets=targets,
        valid_tokens=valid_tokens,
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

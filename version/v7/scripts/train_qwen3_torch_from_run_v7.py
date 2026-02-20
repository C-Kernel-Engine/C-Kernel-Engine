#!/usr/bin/env python3
"""
Train a Qwen-like PyTorch model directly from a v7 run-dir weights snapshot.

This gives an apples-to-apples PyTorch baseline for generated-runtime training
because it uses the same run_dir weights manifest/bump tensors and the same
Qwen-style block structure (RMSNorm, Q/K/V, GQA attention, SwiGLU MLP).
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from oracle_snapshot_torch_v7 import _apply_affine, _apply_head_rms_norm, _apply_rope


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _load_text(prompt: Optional[str], data_path: Optional[Path]) -> str:
    if data_path is not None:
        return data_path.read_text(encoding="utf-8")
    if prompt is None:
        raise ValueError("Need either --prompt or --data")
    return str(prompt)


def _load_token_file(path: Path) -> List[int]:
    ids: List[int] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s:
            continue
        ids.append(int(s))
    if len(ids) < 2:
        raise ValueError("Token file must contain at least 2 token ids")
    return ids


def _build_batches_from_text(text: str, total_tokens: int, seq_len: int, vocab: int) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    data = (text or "").encode("utf-8", errors="ignore")
    if len(data) < 2:
        raise ValueError("Training text must encode to at least 2 bytes")
    ids = [int(b) % int(vocab) for b in data]
    needed = int(total_tokens) + 1
    repeats = (needed + len(ids) - 1) // len(ids)
    stream = np.array((ids * repeats)[:needed], dtype=np.int64)
    batches: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for i in range(0, total_tokens - seq_len + 1, seq_len):
        x = torch.from_numpy(stream[i : i + seq_len]).long().view(1, seq_len)
        y = torch.from_numpy(stream[i + 1 : i + seq_len + 1]).long().view(1, seq_len)
        batches.append((x, y))
    return batches


def _build_batches_from_ids(token_ids: Sequence[int], total_tokens: int, seq_len: int, vocab: int) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    ids = [int(x) % int(vocab) for x in token_ids]
    if len(ids) < 2:
        raise ValueError("Need at least 2 token ids")
    needed = int(total_tokens) + 1
    repeats = (needed + len(ids) - 1) // len(ids)
    stream = np.array((ids * repeats)[:needed], dtype=np.int64)
    batches: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for i in range(0, total_tokens - seq_len + 1, seq_len):
        x = torch.from_numpy(stream[i : i + seq_len]).long().view(1, seq_len)
        y = torch.from_numpy(stream[i + 1 : i + seq_len + 1]).long().view(1, seq_len)
        batches.append((x, y))
    return batches


def _manifest_entries_map(manifest: dict) -> Dict[str, dict]:
    entries = manifest.get("entries")
    if not isinstance(entries, list):
        raise ValueError("Invalid manifest: missing entries[]")
    out: Dict[str, dict] = {}
    for row in entries:
        if not isinstance(row, dict):
            continue
        name = row.get("name")
        if isinstance(name, str) and name:
            out[name] = row
    return out


def _load_tensor_from_bump(entry: dict, bump_blob: bytes) -> torch.Tensor:
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
    return torch.from_numpy(arr.astype(np.float32, copy=False))


def _infer_num_layers(entries: Iterable[str]) -> int:
    max_layer = -1
    for name in entries:
        if not name.startswith("layer."):
            continue
        parts = name.split(".")
        if len(parts) < 3:
            continue
        if parts[1].isdigit():
            max_layer = max(max_layer, int(parts[1]))
    return max_layer + 1 if max_layer >= 0 else 0


@dataclass
class LoadedRunWeights:
    tensors: Dict[str, torch.Tensor]
    cfg: dict


def _load_run_weights(run_dir: Path) -> LoadedRunWeights:
    manifest_path = run_dir / "weights_manifest.json"
    bump_path = run_dir / "weights.bump"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")
    if not bump_path.exists():
        raise FileNotFoundError(f"Missing bump: {bump_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cfg = dict(manifest.get("config") or {})
    entries = _manifest_entries_map(manifest)
    bump_blob = bump_path.read_bytes()

    tensors: Dict[str, torch.Tensor] = {}
    for name, entry in entries.items():
        if str(entry.get("dtype", "")).lower() != "fp32":
            continue
        tensors[name] = _load_tensor_from_bump(entry, bump_blob)

    if "num_layers" not in cfg or int(cfg.get("num_layers", 0) or 0) <= 0:
        cfg["num_layers"] = _infer_num_layers(tensors.keys())
    if "embed_dim" not in cfg or int(cfg.get("embed_dim", 0) or 0) <= 0:
        tok = tensors.get("token_emb")
        if tok is not None and tok.ndim == 2:
            cfg["embed_dim"] = int(tok.shape[1])
            cfg["vocab_size"] = int(tok.shape[0])
    if "hidden_size" not in cfg or int(cfg.get("hidden_size", 0) or 0) <= 0:
        w1 = tensors.get("layer.0.w1")
        if w1 is not None and w1.ndim == 2:
            cfg["hidden_size"] = int(max(w1.shape[0], w1.shape[1]) // 2)
    if "num_heads" not in cfg or int(cfg.get("num_heads", 0) or 0) <= 0:
        cfg["num_heads"] = 1
    if "num_kv_heads" not in cfg or int(cfg.get("num_kv_heads", 0) or 0) <= 0:
        cfg["num_kv_heads"] = int(cfg["num_heads"])
    if "head_dim" not in cfg or int(cfg.get("head_dim", 0) or 0) <= 0:
        ed = int(cfg.get("embed_dim", 0) or 0)
        nh = int(cfg.get("num_heads", 1) or 1)
        cfg["head_dim"] = int(ed // max(1, nh))
    if "rope_theta" not in cfg:
        cfg["rope_theta"] = 10000.0

    return LoadedRunWeights(tensors=tensors, cfg=cfg)


class QwenLayerWeights(nn.Module):
    def __init__(self, tensors: Dict[str, torch.Tensor], layer_id: int):
        super().__init__()
        p = f"layer.{layer_id}."
        self.ln1_gamma = nn.Parameter(tensors[p + "ln1_gamma"].clone())
        self.wq = nn.Parameter(tensors[p + "wq"].clone())
        self.wk = nn.Parameter(tensors[p + "wk"].clone())
        self.wv = nn.Parameter(tensors[p + "wv"].clone())
        self.wo = nn.Parameter(tensors[p + "wo"].clone())
        self.bq = nn.Parameter(tensors[p + "bq"].clone()) if (p + "bq") in tensors else None
        self.bk = nn.Parameter(tensors[p + "bk"].clone()) if (p + "bk") in tensors else None
        self.bv = nn.Parameter(tensors[p + "bv"].clone()) if (p + "bv") in tensors else None
        self.bo = nn.Parameter(tensors[p + "bo"].clone()) if (p + "bo") in tensors else None
        self.q_norm = nn.Parameter(tensors[p + "q_norm"].clone()) if (p + "q_norm") in tensors else None
        self.k_norm = nn.Parameter(tensors[p + "k_norm"].clone()) if (p + "k_norm") in tensors else None
        self.ln2_gamma = nn.Parameter(tensors[p + "ln2_gamma"].clone())
        self.w1 = nn.Parameter(tensors[p + "w1"].clone())
        self.b1 = nn.Parameter(tensors[p + "b1"].clone()) if (p + "b1") in tensors else None
        self.w2 = nn.Parameter(tensors[p + "w2"].clone())
        self.b2 = nn.Parameter(tensors[p + "b2"].clone()) if (p + "b2") in tensors else None

    @staticmethod
    def _rms_norm(x: torch.Tensor, gamma: torch.Tensor, eps: float) -> torch.Tensor:
        var = torch.mean(x * x, dim=-1, keepdim=True)
        return x * torch.rsqrt(var + eps) * gamma

    def forward(
        self,
        x: torch.Tensor,
        *,
        eps: float,
        n_heads: int,
        n_kv_heads_cfg: int,
        head_dim: int,
        rope_theta: float,
    ) -> torch.Tensor:
        bsz, tsz, _ = x.shape
        residual = x

        x_norm = self._rms_norm(x, self.ln1_gamma, eps)
        q = _apply_affine(x_norm, self.wq, self.bq)
        k = _apply_affine(x_norm, self.wk, self.bk)
        v = _apply_affine(x_norm, self.wv, self.bv)

        q_expected = max(1, int(n_heads) * int(head_dim))
        kv_expected = max(1, int(n_kv_heads_cfg) * int(head_dim))
        if int(q.shape[-1]) >= q_expected:
            q = q[..., :q_expected]
        if int(k.shape[-1]) >= kv_expected:
            k = k[..., :kv_expected]
        if int(v.shape[-1]) >= kv_expected:
            v = v[..., :kv_expected]

        q_out = int(q.shape[-1])
        k_out = int(k.shape[-1])
        q_heads = n_heads if (n_heads * head_dim) == q_out else max(1, q_out // max(1, head_dim))
        kv_heads_nom = n_kv_heads_cfg if (n_kv_heads_cfg * head_dim) == k_out else max(1, k_out // max(1, head_dim))

        qh = q.view(bsz, tsz, q_heads, head_dim).permute(0, 2, 1, 3).contiguous()
        kh = k.view(bsz, tsz, kv_heads_nom, head_dim).permute(0, 2, 1, 3).contiguous()
        vh = v.view(bsz, tsz, kv_heads_nom, head_dim).permute(0, 2, 1, 3).contiguous()

        qh = _apply_head_rms_norm(qh, self.q_norm, eps)
        kh = _apply_head_rms_norm(kh, self.k_norm, eps)
        qh, kh = _apply_rope(qh, kh, rope_theta)

        scale = 1.0 / math.sqrt(float(max(1, head_dim)))
        hq = int(qh.shape[1])
        hk = int(kh.shape[1])
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

        ctx = ctx.permute(0, 2, 1, 3).contiguous().view(bsz, tsz, -1)
        attn_out = _apply_affine(ctx, self.wo, self.bo)
        x = residual + attn_out

        residual2 = x
        x_norm2 = self._rms_norm(x, self.ln2_gamma, eps)
        gate_up = _apply_affine(x_norm2, self.w1, self.b1)
        if int(gate_up.shape[-1]) % 2 != 0:
            raise ValueError(f"Invalid SwiGLU width: {tuple(gate_up.shape)}")
        half = int(gate_up.shape[-1]) // 2
        gate = gate_up[..., :half]
        up = gate_up[..., half:]
        mlp_hidden = F.silu(gate) * up
        mlp_out = _apply_affine(mlp_hidden, self.w2, self.b2)
        return residual2 + mlp_out


class TorchQwenFromRun(nn.Module):
    def __init__(self, tensors: Dict[str, torch.Tensor], cfg: dict):
        super().__init__()
        self.cfg = dict(cfg)
        self.eps = 1e-5

        self.num_layers = int(self.cfg.get("num_layers", 0) or 0)
        self.embed_dim = int(self.cfg.get("embed_dim", 0) or 0)
        self.num_heads = int(self.cfg.get("num_heads", 1) or 1)
        self.num_kv_heads = int(self.cfg.get("num_kv_heads", self.num_heads) or self.num_heads)
        self.head_dim = int(self.cfg.get("head_dim", 0) or 0)
        self.rope_theta = float(self.cfg.get("rope_theta", 10000.0) or 10000.0)

        self.token_emb = nn.Parameter(tensors["token_emb"].clone())
        self.layers = nn.ModuleList([QwenLayerWeights(tensors, i) for i in range(self.num_layers)])
        self.final_ln_weight = nn.Parameter(tensors["final_ln_weight"].clone())
        self.output_weight = nn.Parameter(tensors["output.weight"].clone()) if "output.weight" in tensors else None

        if self.embed_dim <= 0:
            self.embed_dim = int(self.token_emb.shape[1])
        if self.head_dim <= 0:
            self.head_dim = int(self.embed_dim // max(1, self.num_heads))

    @staticmethod
    def _rms_norm(x: torch.Tensor, gamma: torch.Tensor, eps: float) -> torch.Tensor:
        var = torch.mean(x * x, dim=-1, keepdim=True)
        return x * torch.rsqrt(var + eps) * gamma

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = F.embedding(input_ids, self.token_emb)
        for layer in self.layers:
            x = layer(
                x,
                eps=self.eps,
                n_heads=self.num_heads,
                n_kv_heads_cfg=self.num_kv_heads,
                head_dim=self.head_dim,
                rope_theta=self.rope_theta,
            )
        x = self._rms_norm(x, self.final_ln_weight, self.eps)
        out_w = self.output_weight if self.output_weight is not None else self.token_emb
        return torch.matmul(x, out_w.transpose(0, 1))


def train_qwen_from_run(
    *,
    run_dir: Path,
    text: Optional[str],
    token_ids: Optional[Sequence[int]],
    epochs: int,
    seq_len: int,
    total_tokens: int,
    lr: float,
    seed: int,
    max_grad_norm: float,
    max_steps: int,
    weight_decay: float,
) -> dict:
    loaded = _load_run_weights(run_dir)
    vocab = int(loaded.cfg.get("vocab_size", loaded.tensors["token_emb"].shape[0]))
    _seed_all(seed)

    model = TorchQwenFromRun(loaded.tensors, loaded.cfg)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))

    if token_ids is not None:
        batches = _build_batches_from_ids(token_ids, total_tokens=total_tokens, seq_len=seq_len, vocab=vocab)
    else:
        if text is None:
            raise ValueError("Need either text or token_ids")
        batches = _build_batches_from_text(text, total_tokens=total_tokens, seq_len=seq_len, vocab=vocab)
    if not batches:
        raise ValueError("No batches generated")

    steps = 0
    loss_curve: List[dict] = []
    for epoch_idx in range(int(epochs)):
        for x, y in batches:
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1), reduction="mean")
            loss.backward()
            if max_grad_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            steps += 1
            loss_curve.append({"step": steps, "epoch": epoch_idx + 1, "loss": float(loss.detach().cpu().item())})
            if max_steps > 0 and steps >= max_steps:
                break
        if max_steps > 0 and steps >= max_steps:
            break

    losses = [row["loss"] for row in loss_curve]
    final_loss = float(losses[-1]) if losses else float("nan")
    min_loss = float(min(losses)) if losses else float("nan")
    first_loss = float(losses[0]) if losses else float("nan")

    return {
        "run_dir": str(run_dir),
        "epochs": int(epochs),
        "seq_len": int(seq_len),
        "total_tokens": int(total_tokens),
        "steps": int(steps),
        "lr": float(lr),
        "weight_decay": float(weight_decay),
        "max_grad_norm": float(max_grad_norm),
        "seed": int(seed),
        "vocab_size": int(vocab),
        "num_layers": int(model.num_layers),
        "embed_dim": int(model.embed_dim),
        "num_heads": int(model.num_heads),
        "num_kv_heads": int(model.num_kv_heads),
        "head_dim": int(model.head_dim),
        "rope_theta": float(model.rope_theta),
        "first_loss": first_loss,
        "final_loss": final_loss,
        "min_loss": min_loss,
        "loss_curve": loss_curve,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Train Qwen-like PyTorch model from v7 run-dir weights.")
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--data", type=Path, default=None, help="UTF-8 text file for training stream.")
    ap.add_argument("--token-file", type=Path, default=None, help="Pre-tokenized integer stream file (one id per line).")
    ap.add_argument("--prompt", type=str, default=None, help="Inline training text (if --data is not set).")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--seq-len", type=int, default=8)
    ap.add_argument("--total-tokens", type=int, default=8192)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args()

    if not args.run_dir.exists():
        raise SystemExit(f"ERROR: run-dir not found: {args.run_dir}")
    if args.epochs < 1:
        raise ValueError("--epochs must be >= 1")
    if args.seq_len < 1:
        raise ValueError("--seq-len must be >= 1")
    if args.total_tokens < args.seq_len + 1:
        raise ValueError("need total_tokens >= seq_len + 1")

    text: Optional[str] = None
    token_ids: Optional[List[int]] = None
    if args.token_file is not None:
        token_ids = _load_token_file(args.token_file)
    else:
        text = _load_text(args.prompt, args.data)
    summary = train_qwen_from_run(
        run_dir=args.run_dir,
        text=text,
        token_ids=token_ids,
        epochs=int(args.epochs),
        seq_len=int(args.seq_len),
        total_tokens=int(args.total_tokens),
        lr=float(args.lr),
        seed=int(args.seed),
        max_grad_norm=float(args.max_grad_norm),
        max_steps=int(args.max_steps),
        weight_decay=float(args.weight_decay),
    )

    print("=" * 100)
    print("v7 QWEN-LIKE TORCH TRAINING (run-dir sourced)")
    print("=" * 100)
    print(
        f"run={summary['run_dir']} layers={summary['num_layers']} d_model={summary['embed_dim']} "
        f"heads={summary['num_heads']} kv_heads={summary['num_kv_heads']} head_dim={summary['head_dim']}"
    )
    print(
        f"epochs={summary['epochs']} seq_len={summary['seq_len']} total_tokens={summary['total_tokens']} "
        f"steps={summary['steps']} lr={summary['lr']} wd={summary['weight_decay']} max_grad_norm={summary['max_grad_norm']}"
    )
    print(
        f"loss: first={summary['first_loss']:.6f} final={summary['final_loss']:.6f} min={summary['min_loss']:.6f} "
        f"ppl_final={math.exp(summary['final_loss']):.6f}"
    )
    print("=" * 100)

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"JSON: {args.json_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

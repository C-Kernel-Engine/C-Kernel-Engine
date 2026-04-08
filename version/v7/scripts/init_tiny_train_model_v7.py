#!/usr/bin/env python3
"""
init_tiny_train_model_v7.py

Operator-friendly bootstrap initializer for v7 training experiments.

Creates:
- weights.bump          (contiguous fp32 tensor blob)
- weights_manifest.json (entries + config + training kernel policy metadata)
- train_init_config.json (human-readable run config)

This is intentionally tiny-model oriented and deterministic.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
V7_ROOT = SCRIPT_DIR.parent
TEMPLATES_DIR = V7_ROOT / "templates"
TEMPLATE_ALIASES: dict[str, str] = {
    "gemma": "gemma3",
    "gemma3": "gemma3",
    "llama": "llama",
    "nanbeige": "nanbeige",
    "qwen2": "qwen2",
    "qwen3": "qwen3",
    "qwen35": "qwen35",
}


def _align(n: int, a: int) -> int:
    return (n + (a - 1)) & ~(a - 1)


def _resolve_template_name(name: str) -> str:
    text = str(name or "").strip().lower()
    if not text:
        return "qwen3"
    return TEMPLATE_ALIASES.get(text, text)


def _append_tensor(
    blob: bytearray,
    entries: List[Dict],
    name: str,
    arr: np.ndarray,
    align: int = 64,
) -> None:
    assert arr.dtype == np.float32
    off = _align(len(blob), align)
    if off > len(blob):
        blob.extend(b"\x00" * (off - len(blob)))
    raw = arr.tobytes(order="C")
    blob.extend(raw)
    entries.append(
        {
            "name": name,
            "offset": off,
            "size": len(raw),
            "dtype": "fp32",
            "shape": list(arr.shape),
        }
    )


def _fan_in_out(shape: Tuple[int, ...]) -> Tuple[int, int]:
    if len(shape) == 0:
        return 1, 1
    if len(shape) == 1:
        n = int(shape[0]) if int(shape[0]) > 0 else 1
        return n, n
    fan_in = int(shape[-2]) if int(shape[-2]) > 0 else 1
    fan_out = int(shape[-1]) if int(shape[-1]) > 0 else 1
    return fan_in, fan_out


def _init_weight(
    rng: np.random.Generator,
    shape: Tuple[int, ...],
    init: str,
) -> np.ndarray:
    init = str(init or "normal_0p02").lower()
    fan_in, fan_out = _fan_in_out(shape)

    if init == "zeros":
        return np.zeros(shape, dtype=np.float32)
    if init == "xavier_uniform":
        limit = math.sqrt(6.0 / float(fan_in + fan_out))
        return rng.uniform(-limit, limit, size=shape).astype(np.float32)
    if init == "xavier_normal":
        std = math.sqrt(2.0 / float(fan_in + fan_out))
        return (rng.standard_normal(shape).astype(np.float32) * np.float32(std)).astype(np.float32)
    if init == "kaiming_uniform":
        limit = math.sqrt(6.0 / float(fan_in))
        return rng.uniform(-limit, limit, size=shape).astype(np.float32)
    # default: "normal_0p02"
    return (rng.standard_normal(shape).astype(np.float32) * np.float32(0.02)).astype(np.float32)


def _qwen35_layer_kinds(n_layers: int) -> List[str]:
    kinds: List[str] = []
    for layer in range(max(0, int(n_layers))):
        kinds.append("full_attention" if ((layer + 1) % 4 == 0) else "recurrent")
    return kinds


def _qwen35_dims(embed_dim: int, hidden_dim: int, num_heads: int, head_dim: int) -> Dict[str, int]:
    ssm_state_size = max(1, head_dim)
    ssm_group_count = max(1, hidden_dim // max(1, ssm_state_size))
    q_dim = ssm_state_size * ssm_group_count
    gate_dim = max(1, num_heads)
    ssm_inner_size = max(gate_dim, hidden_dim)
    recurrent_head_dim = max(1, ssm_inner_size // gate_dim)
    ssm_inner_size = recurrent_head_dim * gate_dim
    return {
        "attn_out_dim": embed_dim,
        "q_gate_proj_dim": 2 * embed_dim,
        "attn_gate_dim": embed_dim,
        "ssm_state_size": ssm_state_size,
        "ssm_group_count": ssm_group_count,
        "ssm_time_step_rank": gate_dim,
        "ssm_inner_size": ssm_inner_size,
        "ssm_conv_kernel": 4,
        "ssm_conv_history": 3,
        "q_dim": q_dim,
        "k_dim": q_dim,
        "v_dim": ssm_inner_size,
        "gate_dim": gate_dim,
        "recurrent_head_dim": recurrent_head_dim,
        "ssm_conv_channels": q_dim + q_dim + ssm_inner_size,
    }


def build_tiny_model(
    out_dir: Path,
    seed: int,
    init: str,
    template_name: str,
    template_doc: Optional[Dict[str, Any]],
    n_layers: int,
    vocab_size: int,
    embed_dim: int,
    hidden_dim: int,
    num_heads: int,
    num_kv_heads: int,
    context_len: int,
    rope_theta: float,
    kernel_policy: str,
    adamw_beta1: float,
    adamw_beta2: float,
    adamw_eps: float,
    adamw_weight_decay: float,
) -> None:
    if embed_dim % num_heads != 0:
        raise ValueError("embed_dim must be divisible by num_heads")
    if num_heads % num_kv_heads != 0:
        raise ValueError("num_heads must be divisible by num_kv_heads")
    if n_layers <= 0:
        raise ValueError("n_layers must be > 0")

    rng = np.random.default_rng(seed)
    head_dim = embed_dim // num_heads
    blob = bytearray()
    entries: List[Dict] = []

    # Global tensors
    _append_tensor(blob, entries, "token_emb", _init_weight(rng, (vocab_size, embed_dim), init))
    _append_tensor(blob, entries, "final_ln_weight", np.ones((embed_dim,), dtype=np.float32))
    _append_tensor(blob, entries, "output.weight", _init_weight(rng, (vocab_size, embed_dim), init))

    # Tiny parity-harness tensors: these map 1:1 to TinyCKModel/TinyTorchModel state_dict keys.
    # They allow deterministic init replay from weights.bump during `cks-v7-run train/sanity/parity`.
    _append_tensor(blob, entries, "tiny.embedding.weight", _init_weight(rng, (vocab_size, embed_dim), init))
    _append_tensor(blob, entries, "tiny.rms_gamma", np.ones((embed_dim,), dtype=np.float32))
    _append_tensor(blob, entries, "tiny.fc1.weight", _init_weight(rng, (2 * hidden_dim, embed_dim), init))
    _append_tensor(blob, entries, "tiny.fc1.bias", np.zeros((2 * hidden_dim,), dtype=np.float32))
    _append_tensor(blob, entries, "tiny.fc2.weight", _init_weight(rng, (vocab_size, hidden_dim), init))
    _append_tensor(blob, entries, "tiny.fc2.bias", np.zeros((vocab_size,), dtype=np.float32))
    qwen35_dims = _qwen35_dims(embed_dim, hidden_dim, num_heads, head_dim) if str(template_name) == "qwen35" else {}
    layer_kinds = _qwen35_layer_kinds(n_layers) if str(template_name) == "qwen35" else []

    for l in range(n_layers):
        prefix = f"layer.{l}"
        if str(template_name) == "qwen35":
            kind = layer_kinds[l]
            _append_tensor(blob, entries, f"{prefix}.attn_norm", np.ones((embed_dim,), dtype=np.float32))
            _append_tensor(blob, entries, f"{prefix}.post_attention_norm", np.ones((embed_dim,), dtype=np.float32))
            _append_tensor(blob, entries, f"{prefix}.ffn_gate", _init_weight(rng, (2 * hidden_dim, embed_dim), init))
            _append_tensor(blob, entries, f"{prefix}.ffn_down", _init_weight(rng, (embed_dim, hidden_dim), init))
            if kind == "recurrent":
                q_dim = int(qwen35_dims["q_dim"])
                k_dim = int(qwen35_dims["k_dim"])
                v_dim = int(qwen35_dims["v_dim"])
                gate_dim = int(qwen35_dims["gate_dim"])
                conv_channels = int(qwen35_dims["ssm_conv_channels"])
                conv_kernel = int(qwen35_dims["ssm_conv_kernel"])
                recurrent_head_dim = int(qwen35_dims["recurrent_head_dim"])
                _append_tensor(blob, entries, f"{prefix}.attn_qkv", _init_weight(rng, (q_dim + k_dim + v_dim, embed_dim), init))
                _append_tensor(blob, entries, f"{prefix}.attn_gate", _init_weight(rng, (v_dim, embed_dim), init))
                _append_tensor(blob, entries, f"{prefix}.ssm_alpha", _init_weight(rng, (gate_dim, embed_dim), init))
                _append_tensor(blob, entries, f"{prefix}.ssm_beta", _init_weight(rng, (gate_dim, embed_dim), init))
                _append_tensor(blob, entries, f"{prefix}.ssm_conv1d", _init_weight(rng, (conv_channels, conv_kernel), init))
                _append_tensor(blob, entries, f"{prefix}.ssm_dt_bias", np.zeros((gate_dim,), dtype=np.float32))
                _append_tensor(blob, entries, f"{prefix}.ssm_a", np.ones((gate_dim,), dtype=np.float32))
                _append_tensor(blob, entries, f"{prefix}.ssm_norm", np.ones((recurrent_head_dim,), dtype=np.float32))
                _append_tensor(blob, entries, f"{prefix}.ssm_out", _init_weight(rng, (embed_dim, v_dim), init))
            else:
                _append_tensor(blob, entries, f"{prefix}.attn_q_gate", _init_weight(rng, (int(qwen35_dims["q_gate_proj_dim"]), embed_dim), init))
                _append_tensor(blob, entries, f"{prefix}.attn_k", _init_weight(rng, (num_kv_heads * head_dim, embed_dim), init))
                _append_tensor(blob, entries, f"{prefix}.attn_v", _init_weight(rng, (num_kv_heads * head_dim, embed_dim), init))
                _append_tensor(blob, entries, f"{prefix}.attn_output", _init_weight(rng, (embed_dim, int(qwen35_dims["attn_out_dim"])), init))
                _append_tensor(blob, entries, f"{prefix}.attn_q_norm", np.ones((head_dim,), dtype=np.float32))
                _append_tensor(blob, entries, f"{prefix}.attn_k_norm", np.ones((head_dim,), dtype=np.float32))
        else:
            _append_tensor(blob, entries, f"{prefix}.ln1_gamma", np.ones((embed_dim,), dtype=np.float32))
            _append_tensor(blob, entries, f"{prefix}.ln2_gamma", np.ones((embed_dim,), dtype=np.float32))

            _append_tensor(blob, entries, f"{prefix}.wq", _init_weight(rng, (embed_dim, embed_dim), init))
            _append_tensor(blob, entries, f"{prefix}.wk", _init_weight(rng, (num_kv_heads * head_dim, embed_dim), init))
            _append_tensor(blob, entries, f"{prefix}.wv", _init_weight(rng, (num_kv_heads * head_dim, embed_dim), init))
            _append_tensor(blob, entries, f"{prefix}.wo", _init_weight(rng, (embed_dim, embed_dim), init))
            _append_tensor(blob, entries, f"{prefix}.bq", np.zeros((embed_dim,), dtype=np.float32))
            _append_tensor(blob, entries, f"{prefix}.bk", np.zeros((num_kv_heads * head_dim,), dtype=np.float32))
            _append_tensor(blob, entries, f"{prefix}.bv", np.zeros((num_kv_heads * head_dim,), dtype=np.float32))
            _append_tensor(blob, entries, f"{prefix}.bo", np.zeros((embed_dim,), dtype=np.float32))

            _append_tensor(blob, entries, f"{prefix}.q_norm", np.ones((head_dim,), dtype=np.float32))
            _append_tensor(blob, entries, f"{prefix}.k_norm", np.ones((head_dim,), dtype=np.float32))

            # SwiGLU path: w1 emits 2*hidden, w2 projects hidden -> embed
            _append_tensor(blob, entries, f"{prefix}.w1", _init_weight(rng, (2 * hidden_dim, embed_dim), init))
            _append_tensor(blob, entries, f"{prefix}.w2", _init_weight(rng, (embed_dim, hidden_dim), init))
            _append_tensor(blob, entries, f"{prefix}.b1", np.zeros((2 * hidden_dim,), dtype=np.float32))
            _append_tensor(blob, entries, f"{prefix}.b2", np.zeros((embed_dim,), dtype=np.float32))

    cfg = {
        "model": str(template_name or "qwen3"),
        "num_layers": n_layers,
        "embed_dim": embed_dim,
        "hidden_size": hidden_dim,
        "intermediate_size": hidden_dim,
        "vocab_size": vocab_size,
        "num_heads": num_heads,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "context_len": context_len,
        "rope_theta": float(rope_theta),
        "dtype": "fp32",
        "training": {
            "enabled": True,
            "kernel_policy": kernel_policy,
            "precision_policy": "fp32_only",
            "optimizer": {
                "name": "adamw",
                "adamw": {
                    "beta1": float(adamw_beta1),
                    "beta2": float(adamw_beta2),
                    "eps": float(adamw_eps),
                    "weight_decay": float(adamw_weight_decay),
                },
            },
            "tiny_parity": {
                "enabled": True,
                "state_tensors": {
                    "embedding.weight": "tiny.embedding.weight",
                    "rms_gamma": "tiny.rms_gamma",
                    "fc1.weight": "tiny.fc1.weight",
                    "fc1.bias": "tiny.fc1.bias",
                    "fc2.weight": "tiny.fc2.weight",
                    "fc2.bias": "tiny.fc2.bias",
                },
                "vocab": vocab_size,
                "d_model": embed_dim,
                "hidden": hidden_dim,
            },
        },
    }
    if str(template_name) == "qwen35":
        cfg.update(
            {
                "attn_out_dim": int(qwen35_dims["attn_out_dim"]),
                "q_gate_proj_dim": int(qwen35_dims["q_gate_proj_dim"]),
                "attn_gate_dim": int(qwen35_dims["attn_gate_dim"]),
                "ssm_state_size": int(qwen35_dims["ssm_state_size"]),
                "ssm_group_count": int(qwen35_dims["ssm_group_count"]),
                "ssm_time_step_rank": int(qwen35_dims["ssm_time_step_rank"]),
                "ssm_inner_size": int(qwen35_dims["ssm_inner_size"]),
                "ssm_conv_kernel": int(qwen35_dims["ssm_conv_kernel"]),
                "ssm_conv_history": int(qwen35_dims["ssm_conv_history"]),
                "q_dim": int(qwen35_dims["q_dim"]),
                "k_dim": int(qwen35_dims["k_dim"]),
                "v_dim": int(qwen35_dims["v_dim"]),
                "gate_dim": int(qwen35_dims["gate_dim"]),
                "recurrent_num_heads": int(qwen35_dims["gate_dim"]),
                "recurrent_head_dim": int(qwen35_dims["recurrent_head_dim"]),
                "ssm_conv_channels": int(qwen35_dims["ssm_conv_channels"]),
                "num_seqs": 1,
                "layer_kinds": list(layer_kinds),
                "hybrid_block_pattern": list(layer_kinds),
                "full_attention_interval": 4,
            }
        )

    manifest = {
        "version": 1,
        "format": "weights_manifest_v7_tiny_init",
        "config": cfg,
        "entries": entries,
    }
    if isinstance(template_doc, dict) and template_doc:
        # Embed explicit template content so later IR stages do not depend on
        # external file paths.
        manifest["template"] = template_doc

    run_cfg = {
        "seed": seed,
        "init": init,
        "template": str(template_name or "qwen3"),
        "kernel_policy": kernel_policy,
        "architecture": {
            "family": "qwen35-hybrid" if str(template_name) == "qwen35" else "qwen3-like",
            "num_layers": n_layers,
            "vocab_size": vocab_size,
            "embed_dim": embed_dim,
            "hidden_dim": hidden_dim,
            "num_heads": num_heads,
            "num_kv_heads": num_kv_heads,
            "context_len": context_len,
            "rope_theta": rope_theta,
        },
        "artifacts": {
            "weights_bump": "weights.bump",
            "weights_manifest": "weights_manifest.json",
            "train_init_config": "train_init_config.json",
        },
        "tiny_parity": {
            "vocab": vocab_size,
            "d_model": embed_dim,
            "hidden": hidden_dim,
            "state_tensors": {
                "embedding.weight": "tiny.embedding.weight",
                "rms_gamma": "tiny.rms_gamma",
                "fc1.weight": "tiny.fc1.weight",
                "fc1.bias": "tiny.fc1.bias",
                "fc2.weight": "tiny.fc2.weight",
                "fc2.bias": "tiny.fc2.bias",
            },
        },
        "optimizer": {
            "name": "adamw",
            "adamw": {
                "beta1": float(adamw_beta1),
                "beta2": float(adamw_beta2),
                "eps": float(adamw_eps),
                "weight_decay": float(adamw_weight_decay),
            },
        },
    }
    if str(template_name) == "qwen35":
        run_cfg["architecture"].update(
            {
                "attn_out_dim": int(qwen35_dims["attn_out_dim"]),
                "q_gate_proj_dim": int(qwen35_dims["q_gate_proj_dim"]),
                "attn_gate_dim": int(qwen35_dims["attn_gate_dim"]),
                "ssm_state_size": int(qwen35_dims["ssm_state_size"]),
                "ssm_group_count": int(qwen35_dims["ssm_group_count"]),
                "ssm_time_step_rank": int(qwen35_dims["ssm_time_step_rank"]),
                "ssm_inner_size": int(qwen35_dims["ssm_inner_size"]),
                "ssm_conv_kernel": int(qwen35_dims["ssm_conv_kernel"]),
                "recurrent_num_heads": int(qwen35_dims["gate_dim"]),
                "recurrent_head_dim": int(qwen35_dims["recurrent_head_dim"]),
                "layer_kinds": list(layer_kinds),
                "full_attention_interval": 4,
            }
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "weights.bump").write_bytes(bytes(blob))
    (out_dir / "weights_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if isinstance(template_doc, dict) and template_doc:
        run_cfg["artifacts"]["template_train"] = "template_train.json"
        (out_dir / "template_train.json").write_text(json.dumps(template_doc, indent=2), encoding="utf-8")
    (out_dir / "train_init_config.json").write_text(json.dumps(run_cfg, indent=2), encoding="utf-8")

    print(f"Created tiny v7 training model at: {out_dir}")
    print(f"  entries={len(entries)}  bump_bytes={len(blob)}")
    print(f"  manifest={out_dir / 'weights_manifest.json'}")
    print(f"  bump={out_dir / 'weights.bump'}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Initialize tiny fp32 v7 training weights.bump + manifest.")
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--init",
        type=str,
        default="normal_0p02",
        choices=["normal_0p02", "xavier_uniform", "xavier_normal", "kaiming_uniform", "zeros"],
        help="Weight initialization policy (default: normal_0p02)",
    )
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--vocab-size", type=int, default=256)
    ap.add_argument("--embed-dim", type=int, default=128)
    ap.add_argument("--hidden-dim", type=int, default=256)
    ap.add_argument("--num-heads", type=int, default=8)
    ap.add_argument("--num-kv-heads", type=int, default=4)
    ap.add_argument("--context-len", type=int, default=128)
    ap.add_argument("--rope-theta", type=float, default=1_000_000.0)
    ap.add_argument("--kernel-policy", type=str, default="fp32_reference_first")
    ap.add_argument("--adamw-beta1", type=float, default=0.9)
    ap.add_argument("--adamw-beta2", type=float, default=0.999)
    ap.add_argument("--adamw-eps", type=float, default=1e-8)
    ap.add_argument("--adamw-weight-decay", type=float, default=0.01)
    ap.add_argument(
        "--template",
        type=str,
        default="qwen3",
        help="Architecture template name (default: qwen3). Built-ins include qwen2, qwen3, qwen35, gemma/gemma3, llama, nanbeige.",
    )
    ap.add_argument(
        "--template-file",
        type=Path,
        default=None,
        help="Optional custom template JSON path. When set, template is embedded into weights_manifest.json.",
    )
    args = ap.parse_args()

    if not (0.0 <= float(args.adamw_beta1) < 1.0):
        raise ValueError("--adamw-beta1 must be in [0, 1)")
    if not (0.0 <= float(args.adamw_beta2) < 1.0):
        raise ValueError("--adamw-beta2 must be in [0, 1)")
    if not (float(args.adamw_eps) > 0.0):
        raise ValueError("--adamw-eps must be > 0")
    if not (float(args.adamw_weight_decay) >= 0.0):
        raise ValueError("--adamw-weight-decay must be >= 0")

    template_name = _resolve_template_name(str(args.template or "qwen3"))
    template_doc: Optional[Dict[str, Any]] = None
    if args.template_file is not None:
        tf = Path(args.template_file)
        if not tf.exists():
            raise FileNotFoundError(f"template file not found: {tf}")
        template_doc = json.loads(tf.read_text(encoding="utf-8"))
    else:
        built_in = TEMPLATES_DIR / f"{template_name}.json"
        if not built_in.exists():
            raise FileNotFoundError(
                f"unknown template '{template_name}' (expected {built_in} or provide --template-file)"
            )
        template_doc = json.loads(built_in.read_text(encoding="utf-8"))

    build_tiny_model(
        out_dir=args.output_dir,
        seed=int(args.seed),
        init=str(args.init),
        template_name=template_name,
        template_doc=template_doc,
        n_layers=int(args.layers),
        vocab_size=int(args.vocab_size),
        embed_dim=int(args.embed_dim),
        hidden_dim=int(args.hidden_dim),
        num_heads=int(args.num_heads),
        num_kv_heads=int(args.num_kv_heads),
        context_len=int(args.context_len),
        rope_theta=float(args.rope_theta),
        kernel_policy=str(args.kernel_policy),
        adamw_beta1=float(args.adamw_beta1),
        adamw_beta2=float(args.adamw_beta2),
        adamw_eps=float(args.adamw_eps),
        adamw_weight_decay=float(args.adamw_weight_decay),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

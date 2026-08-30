from __future__ import annotations

from typing import Optional

from . import nn


def qwen3_tiny(
    *,
    vocab: int = 256,
    dim: int = 128,
    layers: int = 2,
    hidden: int = 256,
    heads: int = 8,
    kv_heads: int = 4,
    context_len: int = 128,
    rope_theta: float = 1_000_000.0,
    init: str = "xavier_uniform",
    dtype: str = "float32",
    name: str = "tiny_qwen3_module_api",
) -> nn.Sequential:
    """Build the qwen-style tiny LM topology supported by the current v7 adapter."""

    if layers <= 0:
        raise ValueError("layers must be > 0")

    blocks = [
        nn.TransformerBlock(
            dim=dim,
            hidden=hidden,
            heads=heads,
            kv_heads=kv_heads,
            context_len=context_len,
            rope_theta=rope_theta,
            activation="swiglu",
            bias=False,
            init=init,
            dtype=dtype,
            name=f"block{index}",
        )
        for index in range(layers)
    ]
    return nn.Sequential(
        nn.Embedding(vocab=vocab, dim=dim, init=init, dtype=dtype, name="tokens"),
        *blocks,
        nn.RMSNorm(dim, dtype=dtype, name="final_norm"),
        nn.Linear(dim, vocab, bias=False, init=init, dtype=dtype, name="lm_head"),
        name=name,
    )


def qwen3_lm(
    *,
    vocab: int,
    dim: int,
    layers: int,
    hidden: int,
    heads: int,
    kv_heads: Optional[int] = None,
    context_len: int = 2048,
    rope_theta: float = 1_000_000.0,
    init: str = "xavier_uniform",
    dtype: str = "float32",
    name: str = "qwen3_lm",
) -> nn.Sequential:
    """Build a qwen-style LM graph for the current template-export path."""

    return qwen3_tiny(
        vocab=vocab,
        dim=dim,
        layers=layers,
        hidden=hidden,
        heads=heads,
        kv_heads=kv_heads or heads,
        context_len=context_len,
        rope_theta=rope_theta,
        init=init,
        dtype=dtype,
        name=name,
    )

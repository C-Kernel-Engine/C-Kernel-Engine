#!/usr/bin/env python3
"""
Run a numpy forward pass on a C-Kernel-Engine v7 Qwen3-like model and export
per-layer, per-head attention matrices to attention.json for the dataset viewer.

Usage:
    # Use probe_report sequences (best for demo)
    python3 version/v7/tools/export_attention.py \\
        ~/.cache/ck-engine-v7/models/train/toy_svg_atoms_ctx512_d64_h128 --probe

    # Single custom prompt
    python3 version/v7/tools/export_attention.py <run_dir> \\
        --prompt "[shape:circle][color:red][size:small]"

    # Limit number of sequences
    python3 version/v7/tools/export_attention.py <run_dir> --probe --max-probes 4

    # Use a specific checkpoint
    python3 version/v7/tools/export_attention.py <run_dir> --probe --checkpoint 300

    # See available tensors
    python3 version/v7/tools/export_attention.py <run_dir> --list-tensors
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np


# ─────────────────────────── weight loader ───────────────────────────────────

def load_weights(run_dir: Path, checkpoint: int | None = None) -> dict[str, np.ndarray]:
    if checkpoint is not None:
        bump_path = run_dir / "checkpoints" / f"weights_step_{checkpoint:08d}.bump"
    else:
        bump_path = run_dir / "weights.bump"
    manifest_path = run_dir / "weights_manifest.json"

    manifest = json.loads(manifest_path.read_text())
    weights = {}
    with open(bump_path, "rb") as f:
        for e in manifest["entries"]:
            if e.get("dtype", "fp32") != "fp32":
                continue
            f.seek(e["offset"])
            raw = f.read(e["size"])
            arr = np.frombuffer(raw, dtype=np.float32).copy()
            weights[e["name"]] = arr.reshape(e["shape"])
    return weights


def load_config(run_dir: Path) -> dict:
    cfg_path = run_dir / "config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"config.json not found in {run_dir}")
    raw = json.loads(cfg_path.read_text())
    arch = raw.get("architecture", raw)
    return {
        "num_layers":   arch["num_layers"],
        "num_heads":    arch["num_heads"],
        "num_kv_heads": arch["num_kv_heads"],
        "embed_dim":    arch["embed_dim"],
        "hidden_dim":   arch["hidden_dim"],
        "vocab_size":   arch["vocab_size"],
        "context_len":  arch.get("context_len", 512),
        "rope_theta":   float(arch.get("rope_theta", 10000.0)),
        "head_dim":     arch["embed_dim"] // arch["num_heads"],
    }


# ─────────────────────────── tokenizer ───────────────────────────────────────

def build_tokenizer(run_dir: Path):
    """
    Returns (encode_fn, decode_fn) using the tokenizer.json in the run dir.
    Supports the HuggingFace BPE tokenizer.json format.
    Uses the `tokenizers` library if available, otherwise greedy-longest-match.
    """
    tok_path = run_dir / "tokenizer.json"
    if not tok_path.exists():
        raise FileNotFoundError(f"tokenizer.json not found in {run_dir}")

    # Try HF tokenizers library first
    try:
        from tokenizers import Tokenizer  # type: ignore
        hf_tok = Tokenizer.from_file(str(tok_path))

        def encode(text: str) -> tuple[list[int], list[str]]:
            enc = hf_tok.encode(text)
            return enc.ids, enc.tokens

        def decode(ids: list[int]) -> str:
            return hf_tok.decode(ids)

        return encode, decode
    except (ImportError, Exception):
        pass

    # Manual greedy longest-match fallback
    raw = json.loads(tok_path.read_text())
    vocab: dict[str, int] = raw.get("model", {}).get("vocab", raw.get("vocab", {}))
    id_to_tok = {v: k for k, v in vocab.items()}

    # Sort vocabulary by length (longest first) for greedy matching
    sorted_vocab = sorted(vocab.keys(), key=len, reverse=True)

    def encode(text: str) -> tuple[list[int], list[str]]:
        ids, toks = [], []
        i = 0
        while i < len(text):
            matched = False
            for tok_str in sorted_vocab:
                if text[i:].startswith(tok_str):
                    ids.append(vocab[tok_str])
                    toks.append(tok_str)
                    i += len(tok_str)
                    matched = True
                    break
            if not matched:
                unk_id = vocab.get("<|unk|>", vocab.get("<unk>", 0))
                ids.append(unk_id)
                toks.append(text[i])
                i += 1
        return ids, toks

    def decode(ids: list[int]) -> str:
        return "".join(id_to_tok.get(i, "<??>") for i in ids)

    return encode, decode


# ─────────────────────────── math primitives ─────────────────────────────────

def rms_norm(x: np.ndarray, gamma: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Root-mean-square layer normalisation."""
    rms = np.sqrt((x ** 2).mean(axis=-1, keepdims=True) + eps)
    return (x / rms) * gamma


def silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def compute_rope(seq_len: int, head_dim: int, theta: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Precompute RoPE sin/cos tables (non-interleaved / LLaMA style).
    Returns cos, sin of shape (seq_len, head_dim//2).
    """
    half = head_dim // 2
    inv_freq = 1.0 / (theta ** (np.arange(0, half, dtype=np.float32) * 2.0 / head_dim))
    positions = np.arange(seq_len, dtype=np.float32)
    angles = np.outer(positions, inv_freq)   # (seq_len, half)
    return np.cos(angles), np.sin(angles)


def apply_rope(x: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
    """
    Apply RoPE to x of shape (L, head_dim).
    cos, sin: (max_len, head_dim//2) — will be sliced to (L, ...).
    """
    L, D = x.shape
    half = D // 2
    x1 = x[:, :half]
    x2 = x[:, half:]
    c, s = cos[:L], sin[:L]
    return np.concatenate([x1 * c - x2 * s, x1 * s + x2 * c], axis=-1)


# ─────────────────────────── forward pass ────────────────────────────────────

def forward(
    token_ids: list[int],
    weights: dict[str, np.ndarray],
    cfg: dict,
) -> tuple[list[list[list[list[float]]]], np.ndarray]:
    """
    Run a teacher-forcing forward pass and collect attention weights.

    Returns:
        attention_maps: list[layer] of list[head] of (L, L) float lists
        logits: (L, vocab_size) array for the final token predictions
    """
    num_layers   = cfg["num_layers"]
    num_heads    = cfg["num_heads"]
    num_kv_heads = cfg["num_kv_heads"]
    head_dim     = cfg["head_dim"]
    groups       = num_heads // num_kv_heads
    rope_theta   = cfg["rope_theta"]

    ids = np.array(token_ids, dtype=np.int32)
    L   = len(ids)

    # Clamp out-of-range ids
    ids = np.clip(ids, 0, weights["token_emb"].shape[0] - 1)

    # Token embedding lookup
    x = weights["token_emb"][ids].astype(np.float32)   # (L, D)

    # Precompute RoPE tables
    rope_cos, rope_sin = compute_rope(L, head_dim, rope_theta)

    attention_maps = []

    for li in range(num_layers):
        p = f"layer.{li}"

        # ── Pre-attention RMSNorm ──────────────────────────────────────────
        x_norm = rms_norm(x, weights[f"{p}.ln1_gamma"])

        # ── QKV projections ───────────────────────────────────────────────
        Q = x_norm @ weights[f"{p}.wq"].T + weights[f"{p}.bq"]  # (L, H*D)
        K = x_norm @ weights[f"{p}.wk"].T + weights[f"{p}.bk"]  # (L, Hkv*D)
        V = x_norm @ weights[f"{p}.wv"].T + weights[f"{p}.bv"]  # (L, Hkv*D)

        Q = Q.reshape(L, num_heads,    head_dim)   # (L, H,   D_h)
        K = K.reshape(L, num_kv_heads, head_dim)   # (L, Hkv, D_h)
        V = V.reshape(L, num_kv_heads, head_dim)   # (L, Hkv, D_h)

        # ── Per-head QK-Norm (Qwen3 style) ────────────────────────────────
        q_g = weights[f"{p}.q_norm"]
        k_g = weights[f"{p}.k_norm"]
        for h in range(num_heads):
            Q[:, h, :] = rms_norm(Q[:, h, :], q_g)
        for h in range(num_kv_heads):
            K[:, h, :] = rms_norm(K[:, h, :], k_g)

        # ── RoPE ──────────────────────────────────────────────────────────
        for h in range(num_heads):
            Q[:, h, :] = apply_rope(Q[:, h, :], rope_cos, rope_sin)
        for h in range(num_kv_heads):
            K[:, h, :] = apply_rope(K[:, h, :], rope_cos, rope_sin)

        # ── GQA attention (causal) ─────────────────────────────────────────
        scale = head_dim ** -0.5
        causal_mask = np.triu(np.full((L, L), -1e9, dtype=np.float32), k=1)

        layer_heads_attn = []
        head_outs = []

        for h in range(num_heads):
            kv_h = h // groups
            scores = Q[:, h, :] @ K[:, kv_h, :].T * scale  # (L, L)
            scores += causal_mask
            attn = softmax(scores, axis=-1)                  # (L, L)
            layer_heads_attn.append(attn.tolist())
            head_outs.append(attn @ V[:, kv_h, :])          # (L, D_h)

        attention_maps.append(layer_heads_attn)

        # ── Attention output projection + residual ─────────────────────────
        attn_out = np.concatenate(head_outs, axis=-1)        # (L, H*D_h)
        attn_out = attn_out @ weights[f"{p}.wo"].T + weights[f"{p}.bo"]
        x = x + attn_out

        # ── Pre-FFN RMSNorm ────────────────────────────────────────────────
        x_norm = rms_norm(x, weights[f"{p}.ln2_gamma"])

        # ── SwiGLU FFN ────────────────────────────────────────────────────
        # w1 stacks gate and up: shape (2*hidden, embed_dim)
        gate_up = x_norm @ weights[f"{p}.w1"].T + weights[f"{p}.b1"]  # (L, 2H)
        hidden = gate_up.shape[1] // 2
        gate, up = gate_up[:, :hidden], gate_up[:, hidden:]
        ffn_out = silu(gate) * up @ weights[f"{p}.w2"].T + weights[f"{p}.b2"]
        x = x + ffn_out

    # Final RMSNorm + LM head
    x = rms_norm(x, weights["final_ln_weight"])
    logits = x @ weights["output.weight"].T                  # (L, vocab)

    return attention_maps, logits


# ─────────────────────────── sequence helpers ────────────────────────────────

def build_sequences_from_probe(run_dir: Path, max_probes: int, encode_fn) -> list[dict]:
    """Load probe_report.json sequences and tokenize them."""
    # Find probe report
    candidates = sorted(run_dir.glob("*probe_report*.json"))
    if not candidates:
        print("WARNING: No probe_report*.json found; falling back to empty list.", file=sys.stderr)
        return []
    report = json.loads(candidates[0].read_text())

    seqs = []
    for i, r in enumerate(report.get("results", [])[:max_probes]):
        # Compose the full sequence: prompt + expected output
        prompt = r.get("prompt", "")
        expected = r.get("expected_svg", r.get("raw_response", ""))
        # Use expected SVG for teacher-forcing (cleaner than raw_response)
        full_text = prompt + expected

        # Prepend BOS
        bos_id = None
        try:
            tok_raw = json.loads((run_dir / "tokenizer.json").read_text())
            vocab = tok_raw.get("model", {}).get("vocab", {})
            bos_id = vocab.get("<|bos|>")
        except Exception:
            pass

        ids, toks = encode_fn(full_text)
        if bos_id is not None:
            ids  = [bos_id] + ids
            toks = ["<|bos|>"] + toks

        seqs.append({
            "id":     f"probe_{i}",
            "label":  r.get("label", f"probe {i}"),
            "split":  r.get("split", "?"),
            "prompt": prompt,
            "tokens": toks,
            "token_ids": ids,
        })

    return seqs


def build_sequences_from_prompts(prompts: list[str], run_dir: Path, encode_fn) -> list[dict]:
    tok_raw = json.loads((run_dir / "tokenizer.json").read_text())
    vocab   = tok_raw.get("model", {}).get("vocab", {})
    bos_id  = vocab.get("<|bos|>")

    seqs = []
    for i, prompt in enumerate(prompts):
        ids, toks = encode_fn(prompt)
        if bos_id is not None:
            ids  = [bos_id] + ids
            toks = ["<|bos|>"] + toks
        seqs.append({
            "id": f"custom_{i}", "label": f"Prompt: {prompt[:40]}",
            "split": "custom", "prompt": prompt,
            "tokens": toks, "token_ids": ids,
        })
    return seqs


# ─────────────────────────── main ────────────────────────────────────────────

def main(argv=None):
    p = argparse.ArgumentParser(
        description="Export attention matrices from a v7 run to JSON for the dataset viewer.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("run_dir")
    p.add_argument("-o", "--output", help="Output path (default: <run_dir>/attention.json)")
    p.add_argument("--probe", action="store_true",
                   help="Use sequences from probe_report.json in the run directory")
    p.add_argument("--max-probes", type=int, default=6,
                   help="Max probe sequences to include (default: 6)")
    p.add_argument("--prompt", action="append", dest="prompts", metavar="TEXT",
                   help="Add a custom prompt sequence (can be repeated)")
    p.add_argument("--checkpoint", type=int, metavar="STEP",
                   help="Load weights from a checkpoint step")
    p.add_argument("--list-tensors", action="store_true")
    args = p.parse_args(argv)

    run_dir = Path(args.run_dir).expanduser().resolve()
    if not run_dir.is_dir():
        print(f"ERROR: {run_dir} is not a directory", file=sys.stderr)
        return 1

    weights = load_weights(run_dir, args.checkpoint)

    if args.list_tensors:
        for name, arr in sorted(weights.items()):
            print(f"  {name:<40} {arr.dtype}  {'×'.join(str(d) for d in arr.shape)}")
        return 0

    cfg = load_config(run_dir)
    print(f"Model: {run_dir.name}")
    print(f"  layers={cfg['num_layers']}  heads={cfg['num_heads']}  "
          f"kv_heads={cfg['num_kv_heads']}  head_dim={cfg['head_dim']}  "
          f"embed_dim={cfg['embed_dim']}")

    encode_fn, _ = build_tokenizer(run_dir)

    sequences: list[dict] = []
    if args.probe:
        sequences += build_sequences_from_probe(run_dir, args.max_probes, encode_fn)
    if args.prompts:
        sequences += build_sequences_from_prompts(args.prompts, run_dir, encode_fn)
    if not sequences:
        print("ERROR: No sequences specified. Use --probe or --prompt.", file=sys.stderr)
        return 1

    results = []
    for seq in sequences:
        ids   = seq["token_ids"]
        toks  = seq["tokens"]
        L     = len(ids)
        print(f"  [{seq['split']:8s}] {seq['label'][:50]:50s}  L={L}")

        attn_maps, logits = forward(ids, weights, cfg)

        # Convert to compact representation
        layers_out = []
        for li, layer_heads in enumerate(attn_maps):
            heads_out = []
            for hi, mat in enumerate(layer_heads):
                # Round to 4 decimal places to keep JSON small
                heads_out.append({
                    "head": hi,
                    "attn": [[round(v, 5) for v in row] for row in mat],
                })
            layers_out.append({"layer": li, "heads": heads_out})

        # Top predicted token at each position (for context)
        top_preds = logits.argmax(axis=-1).tolist()

        results.append({
            "id":        seq["id"],
            "label":     seq["label"],
            "split":     seq["split"],
            "prompt":    seq["prompt"],
            "tokens":    toks,
            "token_ids": ids,
            "top_preds": top_preds,
            "layers":    layers_out,
        })

    out = {
        "format":  "ck-attention.v1",
        "run_id":  run_dir.name,
        "step":    json.loads((run_dir / "weights_manifest.json").read_text()).get("step"),
        "config":  cfg,
        "sequences": results,
    }

    output_path = Path(args.output).expanduser() if args.output else run_dir / "attention.json"
    output_path.write_text(json.dumps(out, separators=(",", ":")))
    size_kb = output_path.stat().st_size / 1024
    print(f"\nWritten → {output_path}  ({size_kb:.1f} KB, {len(results)} sequences)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

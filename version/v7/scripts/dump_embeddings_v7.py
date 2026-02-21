#!/usr/bin/env python3
"""
Dump token embedding vectors to JSON for the v7 IR visualizer Embedding Space Explorer.

Extracts the token embedding weight matrix from a GGUF model file or HuggingFace
checkpoint and writes a JSON file that the visualizer can load directly.

Usage:
    # From a GGUF file:
    python3 dump_embeddings_v7.py --gguf build/model.gguf --max-tokens 200

    # From a HuggingFace model:
    python3 dump_embeddings_v7.py --hf Qwen/Qwen3-0.6B --max-tokens 200

    # With a tokenizer for human-readable token names:
    python3 dump_embeddings_v7.py --gguf build/model.gguf --tokenizer build/tokenizer.json

Output: embedding_dump.json (loadable by the IR visualizer Data & Tokenizer tab)
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


def extract_gguf_embeddings(gguf_path: Path, max_tokens: int) -> Optional[Dict[str, Any]]:
    """Extract token_embd.weight from a GGUF file (F32 or F16 tensors)."""
    try:
        import numpy as np
    except ImportError:
        print("numpy required for GGUF parsing: pip install numpy", file=sys.stderr)
        return None

    try:
        # Try using llama-cpp-python's gguf reader if available
        from gguf import GGUFReader  # type: ignore
        reader = GGUFReader(str(gguf_path))
        for tensor in reader.tensors:
            name = tensor.name
            if "token_embd" in name or "tok_embeddings" in name or "wte" in name:
                data = tensor.data
                if hasattr(data, 'numpy'):
                    data = data.numpy()
                data = np.array(data, dtype=np.float32)
                if data.ndim == 1:
                    # Reshape: try to infer from tensor shape metadata
                    shape = list(tensor.shape)
                    if len(shape) == 2:
                        data = data.reshape(shape)
                    else:
                        print(f"Cannot reshape 1D tensor with shapes: {shape}", file=sys.stderr)
                        return None

                n_tokens = min(data.shape[0], max_tokens)
                dim = data.shape[1]
                tokens = []
                for i in range(n_tokens):
                    tokens.append({
                        "id": i,
                        "text": f"tok_{i}",
                        "vector": data[i].tolist(),
                    })
                return {"tokens": tokens, "dim": int(dim), "source": f"gguf:{gguf_path.name}"}
        print("No embedding tensor found in GGUF file", file=sys.stderr)
        return None
    except ImportError:
        print("Install gguf package for GGUF support: pip install gguf", file=sys.stderr)
        return None


def extract_hf_embeddings(model_id: str, max_tokens: int) -> Optional[Dict[str, Any]]:
    """Extract embedding weights from a HuggingFace model."""
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError:
        print("torch and transformers required: pip install torch transformers", file=sys.stderr)
        return None

    print(f"Loading HuggingFace model: {model_id}")
    model = AutoModel.from_pretrained(model_id, trust_remote_code=True)
    emb_layer = model.get_input_embeddings()
    if emb_layer is None:
        print("Could not find input embeddings layer", file=sys.stderr)
        return None

    weights = emb_layer.weight.detach().float().cpu().numpy()
    n_tokens = min(weights.shape[0], max_tokens)
    dim = weights.shape[1]

    # Try to get token text from tokenizer
    tok_texts: Dict[int, str] = {}
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        vocab = tokenizer.get_vocab()
        inv_vocab = {v: k for k, v in vocab.items()}
        for i in range(n_tokens):
            if i in inv_vocab:
                tok_texts[i] = inv_vocab[i]
    except Exception:
        pass

    tokens = []
    for i in range(n_tokens):
        tokens.append({
            "id": i,
            "text": tok_texts.get(i, f"tok_{i}"),
            "vector": weights[i].tolist(),
            "special": i < 4,  # Typical: unk, bos, eos, pad
        })

    return {"tokens": tokens, "dim": int(dim), "source": f"hf:{model_id}"}


def add_tokenizer_labels(data: Dict[str, Any], tokenizer_path: Path) -> None:
    """Override token text labels from a tokenizer.json file."""
    try:
        tok_json = json.loads(tokenizer_path.read_text())
    except Exception as e:
        print(f"Warning: could not read tokenizer.json: {e}", file=sys.stderr)
        return

    vocab = {}
    model_section = tok_json.get("model", {})
    if "vocab" in model_section and isinstance(model_section["vocab"], dict):
        vocab = {v: k for k, v in model_section["vocab"].items()}

    # Also check added_tokens
    for at in tok_json.get("added_tokens", []):
        tid = at.get("id")
        content = at.get("content", "")
        if tid is not None and content:
            vocab[tid] = content

    for tok in data.get("tokens", []):
        tid = tok.get("id")
        if tid in vocab:
            tok["text"] = vocab[tid]
            # Mark special tokens
            if vocab[tid].startswith("<|") or vocab[tid].startswith("<s") or vocab[tid] in ("<unk>", "<pad>"):
                tok["special"] = True


def annotate_modes(data: Dict[str, Any], prefill_count: int = 10) -> None:
    """Mark first N tokens as prefill, rest as decode, for visualization coloring."""
    for tok in data.get("tokens", []):
        tid = tok.get("id", 0)
        if tok.get("special"):
            tok["mode"] = ""
        elif tid < prefill_count:
            tok["mode"] = "prefill"
        else:
            tok["mode"] = "decode"


def main() -> int:
    parser = argparse.ArgumentParser(description="Dump token embeddings for v7 IR visualizer")
    parser.add_argument("--gguf", type=Path, help="Path to GGUF model file")
    parser.add_argument("--hf", type=str, help="HuggingFace model ID (e.g., Qwen/Qwen3-0.6B)")
    parser.add_argument("--tokenizer", type=Path, help="Path to tokenizer.json for token labels")
    parser.add_argument("--max-tokens", type=int, default=200, help="Max tokens to dump (default: 200)")
    parser.add_argument("--prefill-count", type=int, default=10, help="First N tokens marked as prefill (default: 10)")
    parser.add_argument("--output", type=Path, default=Path("embedding_dump.json"), help="Output path")
    args = parser.parse_args()

    if not args.gguf and not args.hf:
        parser.error("Provide --gguf or --hf")

    data = None
    if args.gguf:
        data = extract_gguf_embeddings(args.gguf, args.max_tokens)
    elif args.hf:
        data = extract_hf_embeddings(args.hf, args.max_tokens)

    if not data:
        print("Failed to extract embeddings", file=sys.stderr)
        return 1

    # Apply tokenizer labels if provided
    if args.tokenizer:
        add_tokenizer_labels(data, args.tokenizer)

    # Annotate with prefill/decode mode
    annotate_modes(data, args.prefill_count)

    # Write
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(data, f)

    print(f"Wrote {args.output}: {len(data['tokens'])} tokens, {data['dim']}D, source={data['source']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

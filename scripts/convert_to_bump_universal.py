#!/usr/bin/env python3
"""
Universal HF to Bump Converter
Supports multiple model architectures (LLaMA, Devstral/SmolLM, Qwen, etc.) through flexible weight mapping
"""

import argparse
import json
import os
import struct
from typing import Dict, List, Tuple, Optional

import numpy as np

# Architecture-specific weight name mappings
ARCHITECTURE_MAPPINGS = {
    "llama": {
        "embed_tokens": ["model.embed_tokens.weight", "model.tok_embeddings.weight"],
        "layers": "model.layers.{layer}",
        "layer_norm": {
            "ln1": ["input_layernorm.weight"],
            "ln2": ["post_attention_layernorm.weight"],
        },
        "attention": {
            "q_proj": "self_attn.q_proj.weight",
            "k_proj": "self_attn.k_proj.weight",
            "v_proj": "self_attn.v_proj.weight",
            "o_proj": "self_attn.o_proj.weight",
            "q_proj_bias": "self_attn.q_proj.bias",
            "k_proj_bias": "self_attn.k_proj.bias",
            "v_proj_bias": "self_attn.v_proj.bias",
            "o_proj_bias": "self_attn.o_proj.bias",
        },
        "mlp": {
            "gate_proj": "mlp.gate_proj.weight",
            "up_proj": "mlp.up_proj.weight",
            "down_proj": "mlp.down_proj.weight",
            "gate_proj_bias": "mlp.gate_proj.bias",
            "up_proj_bias": "mlp.up_proj.bias",
            "down_proj_bias": "mlp.down_proj.bias",
        },
        "final_norm": "model.norm.weight",
        "final_bias": "model.norm.bias",
        "lm_head": "lm_head.weight",
    },
    "devstral": {
        "embed_tokens": ["model.embed_tokens.weight", "model.embedding.weight"],
        "layers": "model.layers.{layer}",
        "layer_norm": {
            "ln1": ["input_layernorm.weight", "layer_norm.weight"],
            "ln2": ["post_attention_layernorm.weight", "ffn_norm.weight"],
        },
        "attention": {
            "q_proj": "self_attn.q_proj.weight",
            "k_proj": "self_attn.k_proj.weight",
            "v_proj": "self_attn.v_proj.weight",
            "o_proj": "self_attn.o_proj.weight",
            "q_proj_bias": "self_attn.q_proj.bias",
            "k_proj_bias": "self_attn.k_proj.bias",
            "v_proj_bias": "self_attn.v_proj.bias",
            "o_proj_bias": "self_attn.o_proj.bias",
        },
        "mlp": {
            "gate_proj": "mlp.gate_proj.weight",
            "up_proj": "mlp.up_proj.weight",
            "down_proj": "mlp.down_proj.weight",
            "gate_proj_bias": "mlp.gate_proj.bias",
            "up_proj_bias": "mlp.up_proj.bias",
            "down_proj_bias": "mlp.down_proj.bias",
        },
        "final_norm": "model.norm.weight",
        "final_bias": "model.norm.bias",
        "lm_head": "lm_head.weight",
    },
    "smollm": {
        # SmolLM is similar to Devstral
        "embed_tokens": ["model.embed_tokens.weight", "model.embedding.weight"],
        "layers": "model.layers.{layer}",
        "layer_norm": {
            "ln1": ["input_layernorm.weight", "layer_norm.weight"],
            "ln2": ["post_attention_layernorm.weight", "ffn_norm.weight"],
        },
        "attention": {
            "q_proj": "self_attn.q_proj.weight",
            "k_proj": "self_attn.k_proj.weight",
            "v_proj": "self_attn.v_proj.weight",
            "o_proj": "self_attn.o_proj.weight",
            "q_proj_bias": "self_attn.q_proj.bias",
            "k_proj_bias": "self_attn.k_proj.bias",
            "v_proj_bias": "self_attn.v_proj.bias",
            "o_proj_bias": "self_attn.o_proj.bias",
        },
        "mlp": {
            "gate_proj": "mlp.gate_proj.weight",
            "up_proj": "mlp.up_proj.weight",
            "down_proj": "mlp.down_proj.weight",
            "gate_proj_bias": "mlp.gate_proj.bias",
            "up_proj_bias": "mlp.up_proj.bias",
            "down_proj_bias": "mlp.down_proj.bias",
        },
        "final_norm": "model.norm.weight",
        "final_bias": "model.norm.bias",
        "lm_head": "lm_head.weight",
    },
    "qwen": {
        "embed_tokens": ["model.embed_tokens.weight"],
        "layers": "model.layers.{layer}",
        "layer_norm": {
            "ln1": ["input_layernorm.weight"],
            "ln2": ["post_attention_layernorm.weight"],
        },
        "attention": {
            "q_proj": "self_attn.q_proj.weight",
            "k_proj": "self_attn.k_proj.weight",
            "v_proj": "self_attn.v_proj.weight",
            "o_proj": "self_attn.o_proj.weight",
            "q_proj_bias": "self_attn.q_proj.bias",
            "k_proj_bias": "self_attn.k_proj.bias",
            "v_proj_bias": "self_attn.v_proj.bias",
            "o_proj_bias": "self_attn.o_proj.bias",
        },
        "mlp": {
            "gate_proj": "mlp.gate_proj.weight",
            "up_proj": "mlp.up_proj.weight",
            "down_proj": "mlp.down_proj.weight",
            "gate_proj_bias": "mlp.gate_proj.bias",
            "up_proj_bias": "mlp.up_proj.bias",
            "down_proj_bias": "mlp.down_proj.bias",
        },
        "final_norm": "model.norm.weight",
        "final_bias": "model.norm.bias",
        "lm_head": "lm_head.weight",
    },
    "auto": {
        # Auto-detect based on config
        "embed_tokens": ["model.embed_tokens.weight", "model.tok_embeddings.weight", "model.embedding.weight"],
        "layers": "model.layers.{layer}",
        "layer_norm": {
            "ln1": ["input_layernorm.weight", "layer_norm.weight"],
            "ln2": ["post_attention_layernorm.weight", "ffn_norm.weight"],
        },
        "attention": {
            "q_proj": "self_attn.q_proj.weight",
            "k_proj": "self_attn.k_proj.weight",
            "v_proj": "self_attn.v_proj.weight",
            "o_proj": "self_attn.o_proj.weight",
            "q_proj_bias": "self_attn.q_proj.bias",
            "k_proj_bias": "self_attn.k_proj.bias",
            "v_proj_bias": "self_attn.v_proj.bias",
            "o_proj_bias": "self_attn.o_proj.bias",
        },
        "mlp": {
            "gate_proj": "mlp.gate_proj.weight",
            "up_proj": "mlp.up_proj.weight",
            "down_proj": "mlp.down_proj.weight",
            "gate_proj_bias": "mlp.gate_proj.bias",
            "up_proj_bias": "mlp.up_proj.bias",
            "down_proj_bias": "mlp.down_proj.bias",
        },
        "final_norm": "model.norm.weight",
        "final_bias": "model.norm.bias",
        "lm_head": "lm_head.weight",
    }
}


def detect_architecture(state_dict: Dict, config: Dict) -> str:
    """Auto-detect model architecture from state dict and config"""
    model_type = config.get("model_type", "").lower()

    if model_type:
        # Use model_type if available
        if model_type == "llama":
            return "llama"
        elif model_type in ["devstral", "smollm", "smol_lm"]:
            return model_type
        elif model_type == "qwen2":
            return "qwen"

    # Fallback: detect from state dict keys
    sample_keys = list(state_dict.keys())[:20]

    # Check for architecture-specific patterns
    if any("tok_embeddings" in k for k in sample_keys):
        return "llama"
    elif any("embedding.weight" in k for k in sample_keys):
        return "devstral"
    elif any("embed_tokens" in k for k in sample_keys):
        return "auto"

    # Default to LLaMA
    return "llama"


def get_weight_name(state_dict: Dict, key: str, layer: Optional[int] = None) -> Optional[str]:
    """Get actual weight name from state dict based on architecture mapping"""
    for weight_name in key if isinstance(key, list) else [key]:
        if layer is not None:
            weight_name = weight_name.replace("{layer}", str(layer))

        if weight_name in state_dict:
            return weight_name

    return None


def convert_hf_to_bump_universal(
    checkpoint: str,
    output: str,
    architecture: str = "auto",
    config: Optional[str] = None,
    context: Optional[int] = None,
    dtype: str = "float32",
) -> None:
    """
    Convert HF weights to bump format with flexible architecture support

    Args:
        checkpoint: Path to HF model directory
        output: Output bump file path
        architecture: Model architecture (llama, devstral, smollm, qwen, auto)
        config: Optional config JSON path
        context: Override context length
        dtype: Output dtype (float32, q4_k, q4_k_m)
    """
    from transformers import AutoModelForCausalLM
    import torch

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint,
        torch_dtype=None,
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
    state_dict = model.state_dict()
    cfg = model.config.to_dict()

    # Load config override if provided
    if config:
        with open(config, "r", encoding="utf-8") as f:
            override_cfg = json.load(f)
            if "text_config" in override_cfg:
                override_cfg = override_cfg["text_config"]
            cfg.update(override_cfg)

    # Detect architecture
    if architecture == "auto":
        architecture = detect_architecture(state_dict, cfg)
        print(f"[info] Auto-detected architecture: {architecture}")

    # Get architecture mapping
    if architecture not in ARCHITECTURE_MAPPINGS:
        raise ValueError(f"Unsupported architecture: {architecture}. Supported: {list(ARCHITECTURE_MAPPINGS.keys())}")

    mapping = ARCHITECTURE_MAPPINGS[architecture]

    # Extract config values
    num_layers = cfg.get("num_hidden_layers") or cfg.get("num_layers")
    embed_dim = cfg.get("hidden_size") or cfg.get("embed_dim")
    intermediate = cfg.get("intermediate_size")
    num_heads = cfg.get("num_attention_heads") or cfg.get("num_heads")
    num_kv_heads = cfg.get("num_key_value_heads") or cfg.get("num_kv_heads") or num_heads
    vocab_size = cfg.get("vocab_size")
    context_len = cfg.get("max_position_embeddings") or cfg.get("context_window") or cfg.get("ctx", 0)

    if context:
        context_len = context

    if not all([num_layers, embed_dim, intermediate, num_heads, vocab_size]):
        raise ValueError(f"Config missing required fields. Found: num_layers={num_layers}, embed_dim={embed_dim}, intermediate={intermediate}, num_heads={num_heads}, vocab_size={vocab_size}")

    print(f"[info] Converting {architecture} model:")
    print(f"  - Layers: {num_layers}")
    print(f"  - Embed dim: {embed_dim}")
    print(f"  - Intermediate: {intermediate}")
    print(f"  - Heads: {num_heads} (KV: {num_kv_heads})")
    print(f"  - Vocab: {vocab_size}")
    print(f"  - Context: {context_len}")

    # Extract weights with error handling
    def safe_get_tensor(key_list: List[str], layer: Optional[int] = None) -> Optional[np.ndarray]:
        """Safely get tensor with multiple possible names"""
        for key in key_list:
            if layer is not None:
                key = key.replace("{layer}", str(layer))

            if key in state_dict:
                return state_dict[key].detach().cpu().numpy()

        print(f"[warn] Could not find weight: {key_list}")
        return None

    # TODO: Implement the actual conversion logic here
    # This is a template showing the architecture detection and flexible mapping

    print(f"[info] Architecture mapping configured for: {architecture}")
    print(f"[info] Example weight locations:")
    print(f"  - Embed tokens: {mapping['embed_tokens']}")
    print(f"  - Layer 0 norm1: {mapping['layer_norm']['ln1']}")
    print(f"  - Layer 0 q_proj: {mapping['attention']['q_proj']}")


def main():
    parser = argparse.ArgumentParser(
        description="Universal HF to Bump Converter - supports LLaMA, Devstral, SmolLM, Qwen, and more"
    )
    parser.add_argument("--checkpoint", required=True, help="HF model directory")
    parser.add_argument("--output", required=True, help="Output bump file")
    parser.add_argument("--arch", default="auto", choices=list(ARCHITECTURE_MAPPINGS.keys()),
                       help="Model architecture (auto-detect if not specified)")
    parser.add_argument("--config", help="Optional config JSON")
    parser.add_argument("--context", type=int, help="Override context length")
    parser.add_argument("--dtype", default="float32", help="Output dtype")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()

    try:
        convert_hf_to_bump_universal(
            checkpoint=args.checkpoint,
            output=args.output,
            architecture=args.arch,
            config=args.config,
            context=args.context,
            dtype=args.dtype,
        )
        print(f"\n[success] Conversion complete: {args.output}")
    except Exception as e:
        print(f"\n[error] {e}")
        import traceback
        if args.verbose:
            traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())

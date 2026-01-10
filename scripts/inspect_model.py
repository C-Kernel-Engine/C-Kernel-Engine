#!/usr/bin/env python3
"""
Inspect HF Model Structure
Quick diagnostic tool to see weight names and architecture of a model
"""

import argparse
import json
from typing import Dict, List

def inspect_model(checkpoint: str, config_only: bool = False) -> None:
    """Inspect model structure"""
    from transformers import AutoConfig, AutoModelForCausalLM
    import torch

    print(f"\n{'='*80}")
    print(f"Model Inspector: {checkpoint}")
    print(f"{'='*80}\n")

    # Load config
    config = AutoConfig.from_pretrained(checkpoint)
    print(f"[CONFIG]")
    print(f"  Model Type: {config.model_type}")
    print(f"  Hidden Size: {getattr(config, 'hidden_size', 'N/A')}")
    print(f"  Vocab Size: {getattr(config, 'vocab_size', 'N/A')}")
    print(f"  Num Layers: {getattr(config, 'num_hidden_layers', getattr(config, 'num_layers', 'N/A'))}")
    print(f"  Num Heads: {getattr(config, 'num_attention_heads', getattr(config, 'num_heads', 'N/A'))}")
    print(f"  Intermediate Size: {getattr(config, 'intermediate_size', 'N/A')}")
    print(f"  Context Length: {getattr(config, 'max_position_embeddings', getattr(config, 'context_window', 'N/A'))}")
    print(f"  Model Name: {getattr(config, '_name_or_path', 'N/A')}")

    if config_only:
        return

    # Load state dict
    print(f"\n[LOADING STATE DICT...]")
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint,
        torch_dtype=None,
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
    state_dict = model.state_dict()

    print(f"\n[WEIGHT STATS]")
    print(f"  Total weights: {len(state_dict)}")

    # Group weights by pattern
    patterns = {}
    for key in sorted(state_dict.keys()):
        # Extract pattern (remove layer numbers, dimensions)
        parts = key.split(".")
        pattern = ".".join(p for p in parts if not (p.isdigit() or p.startswith("[")))

        if pattern not in patterns:
            patterns[pattern] = []
        patterns[pattern].append(key)

    print(f"\n[WEIGHT PATTERNS]")
    for pattern in sorted(patterns.keys()):
        count = len(patterns[pattern])
        print(f"  {pattern:60} x{count}")

    # Show sample weights from each group
    print(f"\n[SAMPLE WEIGHTS]")
    shown_patterns = set()
    for key in sorted(state_dict.keys()):
        pattern = ".".join(key.split(".")[:-1])  # Remove last part (weight/bias)
        if pattern in shown_patterns:
            continue
        shown_patterns.add(pattern)

        shape = state_dict[key].shape
        print(f"  {key:60} {str(shape):20}")

    # Try to identify architecture
    print(f"\n[ARCHITECTURE DETECTION]")
    sample_keys = list(state_dict.keys())[:20]

    architecture_clues = {
        "LLaMA": ["tok_embeddings", "model.norm"],
        "Devstral/SmolLM": ["model.embedding", "model.layers"],
        "Qwen": ["model.embed_tokens", "model.layers"],
        "GPT-NeoX": ["gpt_neox"],
        "GPT-J": ["transformer.h"],
        "MPT": ["wte", "blocks"],
    }

    detected = []
    for arch, clues in architecture_clues.items():
        if any(any(clue in key for clue in clues) for key in sample_keys):
            detected.append(arch)

    if detected:
        print(f"  Detected: {', '.join(detected)}")
    else:
        print(f"  Could not auto-detect (check model_type in config)")

    # Suggest mapping
    print(f"\n[SUGGESTED MAPPING]")
    if config.model_type:
        arch = config.model_type.lower()
        if arch in ["llama", "devstral", "smollm", "qwen2"]:
            print(f"  Use --arch {arch}")
        else:
            print(f"  Model type: {arch} (not in known mappings)")

    print(f"\n{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description="Inspect HF model structure")
    parser.add_argument("--checkpoint", required=True, help="HF model directory")
    parser.add_argument("--config-only", action="store_true", help="Show config only")
    args = parser.parse_args()

    inspect_model(args.checkpoint, config_only=args.config_only)


if __name__ == "__main__":
    main()

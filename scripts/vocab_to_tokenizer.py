#!/usr/bin/env python3
"""
Convert qwen_vocab.json to tokenizer.json format for C-Kernel-Engine tokenizer.

Usage:
    python scripts/vocab_to_tokenizer.py --input qwen_vocab.json --output tokenizer.json
"""

import json
import argparse
import sys

def convert_vocab_to_tokenizer(input_path, output_path, model_name="gpt2"):
    """Convert simple vocab array to HuggingFace tokenizer.json format."""

    # Load vocab
    with open(input_path, 'r') as f:
        data = json.load(f)

    tokens = data.get('tokens', [])
    special = data.get('special_tokens', {})

    vocab_size = len(tokens)
    print(f"Loaded {vocab_size} tokens")

    # Build vocab dict (token -> id)
    vocab = {}
    for i, token in enumerate(tokens):
        vocab[token] = i

    # Get special token IDs
    bos_id = special.get('bos', 1)
    eos_id = special.get('eos', 2)
    unk_id = special.get('unk', 0)
    pad_id = special.get('pad', bos_id)  # Usually same as bos for Qwen

    print(f"Special tokens: BOS={bos_id}, EOS={eos_id}, UNK={unk_id}, PAD={pad_id}")

    # Build merges list (simplified - just use common pairs)
    # For a real tokenizer, we'd compute optimal merges
    # For now, we'll skip merges and let tokenizer use greedy matching

    # Create minimal merges for common patterns
    merges = []

    # Common byte pair patterns for typical text
    # These would normally be computed from training data
    common_pairs = [
        # English common pairs
        ("t", "h", "th"),
        ("h", "e", "he"),
        ("e", "r", "er"),
        ("r", "e", "re"),
        ("i", "n", "in"),
        ("n", "g", "ng"),
        ("o", "u", "ou"),
        ("u", "t", "ut"),
        ("s", "t", "st"),
        ("a", "n", "an"),
        ("a", "nd", "and"),
        ("t", "he", "the"),
        ("i", "t", "it"),
        ("t", "o", "to"),
    ]

    # Add merges for pairs that exist in vocab
    for t1, t2, merged in common_pairs:
        if t1 in vocab and t2 in vocab:
            mid = vocab.get(merged, -1)
            if mid >= 0:
                merges.append(f"{t1} {t2}")

    print(f"Generated {len(merges)} merges")

    # Build output structure
    output = {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": [
            {
                "id": unk_id,
                "content": "<|endoftext|>",
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "special": True
            },
            {
                "id": bos_id,
                "content": "<|im_start|>",
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "special": True
            },
            {
                "id": eos_id,
                "content": "<|im_end|>",
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "special": True
            }
        ],
        "normalizer": {
            "type": "Sequence",
            "add_prefix_space": False,
            "lowercase": False,
            "strip_accents": None
        },
        "pre_tokenizer": {
            "type": "BytesLevel",
            "add_prefix_space": False,
            "use_regex": True
        },
        "post_processor": {
            "type": "TemplateProcessing",
            "single": [
                {
                    "id": bos_id,
                    "type": "SpecialToken",
                    "pattern": {"id": f"<|im_start|>"}
                },
                {
                    "id": "[[ID]]",
                    "type": "TokenType",
                    "pattern": {"id": "[[ID]]"}
                },
                {
                    "id": eos_id,
                    "type": "SpecialToken",
                    "pattern": {"id": f"<|im_end|>"}
                }
            ],
            "pair": [
                {
                    "id": bos_id,
                    "type": "SpecialToken",
                    "pattern": {"id": f"<|im_start|>"}
                },
                {
                    "id": "[[ID]]",
                    "type": "TokenType",
                    "pattern": {"id": "[[ID]]"}
                },
                {
                    "id": bos_id,
                    "type": "SpecialToken",
                    "pattern": {"id": f"<|im_start|>"}
                },
                {
                    "id": "[[ID]]",
                    "type": "TokenType",
                    "pattern": {"id": "[[ID]]"}
                },
                {
                    "id": eos_id,
                    "type": "SpecialToken",
                    "pattern": {"id": f"<|im_end|>"}
                }
            ],
            "special_tokens": {
                "bos": f"<|im_start|>",
                "eos": f"<|im_end|>"
            }
        },
        "decoder": {
            "type": "Sequence",
            "add_prefix_space": True,
            "cleanup": True
        },
        "model": {
            "type": "BPE",
            "dropout": None,
            "unk_token": "<|endoftext|>",
            "beginning_of_word_marker": None,
            "end_of_word_marker": None,
            "fuse_unk": False,
            "byte_fallback": False,
            "vocab": vocab,
            "merges": merges
        }
    }

    # Write output
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"Written tokenizer.json to {output_path}")
    return True

def main():
    parser = argparse.ArgumentParser(description="Convert vocab to tokenizer.json")
    parser.add_argument("--input", "-i", required=True, help="Input vocab JSON file")
    parser.add_argument("--output", "-o", required=True, help="Output tokenizer.json file")
    parser.add_argument("--model", default="gpt2", help="Model type")

    args = parser.parse_args()

    if not convert_vocab_to_tokenizer(args.input, args.output, args.model):
        sys.exit(1)

if __name__ == "__main__":
    main()

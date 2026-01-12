#!/usr/bin/env python3
"""
Chat template utilities for C-Kernel-Engine.

Provides:
- Template definitions for different model families
- Template detection from model config/tokenizer
- Serialization/deserialization for bump files
- Template application helpers
"""
import json
import hashlib
from pathlib import Path
from typing import Dict, Optional, Any


# ═══════════════════════════════════════════════════════════════════════
# Template Definitions
# ═══════════════════════════════════════════════════════════════════════

CHAT_TEMPLATES = {
    "qwen": {
        "template": (
            "<|im_start|>system\n"
            "{system}"
            "<|im_end|>\n"
            "<|im_start|>user\n"
            "{prompt}"
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
        ),
        "default_system": "You are a helpful AI assistant.",
    },
    "llama3": {
        "template": (
            "<|start_header_id|>system<|end_header_id|>\n\n"
            "{system}"
            "<|eot_id|>\n"
            "<|start_header_id|>user<|end_header_id|>\n\n"
            "{prompt}"
            "<|eot_id|>\n"
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
        ),
        "default_system": "You are a helpful, honest assistant.",
    },
    "llama2": {
        "template": (
            "[INST] <<SYS>>\n"
            "{system}\n"
            "<</SYS>>\n\n"
            "{prompt}"
            " [/INST]"
        ),
        "default_system": "You are a helpful and honest assistant.",
    },
    "mistral": {
        "template": (
            "[INST] <<SYS>>\n"
            "{system}\n"
            "<</SYS>>\n\n"
            "{prompt}"
            " [/INST]"
        ),
        "default_system": "You are a helpful AI assistant.",
    },
    "gemma": {
        "template": (
            "<start_of_turn>system\n"
            "{system}"
            "<end_of_turn>\n"
            "<start_of_turn>user\n"
            "{prompt}"
            "<end_of_turn>\n"
            "<start_of_turn>model\n"
        ),
        "default_system": "You are a helpful AI assistant.",
    },
    "phi3": {
        "template": (
            "<|system|>\n"
            "{system}"
            "<|end|>\n"
            "<|user|>\n"
            "{prompt}"
            "<|end|>\n"
            "<|assistant|>\n"
        ),
        "default_system": "You are a helpful AI assistant.",
    },
    "smollm": {
        "template": (
            "<|im_start|>system\n"
            "{system}"
            "<|im_end|>\n"
            "<|im_start|>user\n"
            "{prompt}"
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
        ),
        "default_system": "You are a helpful AI assistant.",
    },
    "none": {
        "template": "{prompt}",
        "default_system": "",
    },
}


# ═══════════════════════════════════════════════════════════════════════
# Template Detection
# ═══════════════════════════════════════════════════════════════════════

def detect_template_from_model_name(model_name: str) -> Dict[str, str]:
    """Detect template from model name string."""
    model_lower = model_name.lower()

    # Direct model family detection
    if "qwen" in model_lower:
        return CHAT_TEMPLATES["qwen"]
    elif "llama-3" in model_lower or "llama3" in model_lower:
        return CHAT_TEMPLATES["llama3"]
    elif "llama-2" in model_lower or "llama2" in model_lower:
        return CHAT_TEMPLATES["llama2"]
    elif "mistral" in model_lower or "mixtral" in model_lower:
        return CHAT_TEMPLATES["mistral"]
    elif "gemma" in model_lower:
        return CHAT_TEMPLATES["gemma"]
    elif "phi-3" in model_lower or "phi3" in model_lower:
        return CHAT_TEMPLATES["phi3"]
    elif "smollm" in model_lower:
        return CHAT_TEMPLATES["smollm"]

    return CHAT_TEMPLATES["none"]


def detect_template_from_tokenizer(tokenizer_json: Path) -> Dict[str, str]:
    """Detect template from tokenizer.json content."""
    try:
        with open(tokenizer_json, 'r', encoding='utf-8') as f:
            data = json.load(f)

        vocab = data.get('model', {}).get('vocab', {})

        # Check for ChatML tokens (Qwen)
        if '<|im_start|>' in vocab or '<|im_end|>' in vocab:
            return CHAT_TEMPLATES["qwen"]

        # Check for Llama 3 tokens
        if '<|start_header_id|>' in vocab:
            return CHAT_TEMPLATES["llama3"]

        # Check for Gemma tokens
        if '<start_of_turn>' in vocab:
            return CHAT_TEMPLATES["gemma"]

    except Exception:
        pass

    return CHAT_TEMPLATES["none"]


def detect_template(config_json: Path, tokenizer_json: Optional[Path] = None) -> Dict[str, str]:
    """
    Detect chat template from model configuration.

    Args:
        config_json: Path to config.json
        tokenizer_json: Optional path to tokenizer.json

    Returns:
        Dict with 'template', 'default_system', 'type'
    """
    # Try config.json first
    try:
        with open(config_json, 'r', encoding='utf-8') as f:
            config = json.load(f)

        # Some models store template in config
        if 'chat_template' in config:
            template_str = config['chat_template']
            if isinstance(template_str, str):
                return {
                    "template": template_str,
                    "default_system": config.get('default_system_prompt', ""),
                    "type": "custom"
                }

        # Try to get model name
        model_name = config.get('model_type') or config.get('name') or ""
        if model_name:
            result = detect_template_from_model_name(model_name)
            if result["template"] != CHAT_TEMPLATES["none"]["template"]:
                return {**result, "type": "detected"}
    except Exception:
        pass

    # Try tokenizer.json
    if tokenizer_json and tokenizer_json.exists():
        result = detect_template_from_tokenizer(tokenizer_json)
        if result["template"] != CHAT_TEMPLATES["none"]["template"]:
            return {**result, "type": "detected"}

    # Default: no template
    return {**CHAT_TEMPLATES["none"], "type": "none"}


# ═══════════════════════════════════════════════════════════════════════
# EOS Token Detection
# ═══════════════════════════════════════════════════════════════════════

def find_eos_token_id(tokenizer_json: Path, eos_token: str = None) -> int:
    """Find the token ID for an EOS token in tokenizer.json."""
    try:
        with open(tokenizer_json, 'r', encoding='utf-8') as f:
            data = json.load(f)

        vocab = data.get('model', {}).get('vocab', {})

        # Try specified token first
        if eos_token:
            if eos_token in vocab:
                return vocab[eos_token]

        # Try common EOS token names
        eos_names = [
            "<|im_end|>",
            "<|endoftext|>",
            "</s>",
            "<|eot_id|>",
            "<eos>",
            "<|end|>",
        ]

        for name in eos_names:
            if name in vocab:
                return vocab[name]

    except Exception:
        pass

    return -1  # Not found


def find_stop_token_ids(tokenizer_json: Path) -> list:
    """Find all stop token IDs from tokenizer.json.

    Returns a list of token IDs that should stop generation.
    """
    stop_names = [
        "<|im_end|>",
        "<|eot_id|>",
        "</s>",
        "<|endoftext|>",
        "<|end|>",
        "<eos>",
        "<|stop|>",
    ]

    try:
        with open(tokenizer_json, 'r', encoding='utf-8') as f:
            data = json.load(f)

        vocab = data.get('model', {}).get('vocab', {})
        stop_ids = []

        for name in stop_names:
            if name in vocab:
                token_id = vocab[name]
                if token_id not in stop_ids:
                    stop_ids.append(token_id)

        return stop_ids
    except Exception:
        return []


def get_vocab_size(tokenizer_json: Path) -> int:
    """Get vocab size from tokenizer.json."""
    try:
        with open(tokenizer_json, 'r', encoding='utf-8') as f:
            data = json.load(f)

        vocab = data.get('model', {}).get('vocab', {})
        return len(vocab)
    except Exception:
        return 0


# ═══════════════════════════════════════════════════════════════════════
# Special Token Detection (from transformers or known model families)
# ═══════════════════════════════════════════════════════════════════════

# Known special tokens for different model families (token_id, token_str)
# These are added dynamically by tokenizers and not in tokenizer.json
SPECIAL_TOKENS_QWEN = {
    "eos_token_id": 151645,  # <|im_end|>
    "bos_token_id": None,
    "stop_token_ids": [151645],  # <|im_end|>
}

SPECIAL_TOKENS_LLAMA3 = {
    "eos_token_id": 128009,  # <|eot_id|>
    "bos_token_id": 128000,  # <|start_header_id|>
    "stop_token_ids": [128009, 128001],  # <|eot_id|>, <|end_header_id|>
}

SPECIAL_TOKENS_LLAMA2 = {
    "eos_token_id": 2,  # </s>
    "bos_token_id": 1,  # <s>
    "stop_token_ids": [2],  # </s>
}

SPECIAL_TOKENS_MISTRAL = {
    "eos_token_id": 2,  # </s>
    "bos_token_id": 1,  # <s>
    "stop_token_ids": [2],  # </s>
}

SPECIAL_TOKENS_GEMMA = {
    "eos_token_id": 1,  # <eos>
    "bos_token_id": None,
    "stop_token_ids": [1],  # <eos>
}

SPECIAL_TOKENS_PHI3 = {
    "eos_token_id": 3,  # <|end|>
    "bos_token_id": None,
    "stop_token_ids": [3],  # <|end|>
}

SPECIAL_TOKENS_SMOLLM = {
    "eos_token_id": 0,  # <|endoftext|> (usually 0 or 1)
    "bos_token_id": None,
    "stop_token_ids": [0],
}

# Mapping model family names to special tokens
MODEL_FAMILY_SPECIAL_TOKENS = {
    "qwen": SPECIAL_TOKENS_QWEN,
    "qwen2": SPECIAL_TOKENS_QWEN,
    "llama3": SPECIAL_TOKENS_LLAMA3,
    "llama-3": SPECIAL_TOKENS_LLAMA3,
    "llama2": SPECIAL_TOKENS_LLAMA2,
    "llama-2": SPECIAL_TOKENS_LLAMA2,
    "mistral": SPECIAL_TOKENS_MISTRAL,
    "mixtral": SPECIAL_TOKENS_MISTRAL,
    "gemma": SPECIAL_TOKENS_GEMMA,
    "phi3": SPECIAL_TOKENS_PHI3,
    "smollm": SPECIAL_TOKENS_SMOLLM,
}


def get_special_tokens_from_hf(model_name_or_path: str) -> dict:
    """
    Get special token IDs from HuggingFace transformers library.

    Returns dict with eos_token_id, bos_token_id, stop_token_ids, vocab_size.
    Returns empty dict if transformers unavailable or fails.
    """
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)

        result = {}

        # Get eos_token_id
        if hasattr(tokenizer, 'eos_token_id') and tokenizer.eos_token_id:
            result["eos_token_id"] = int(tokenizer.eos_token_id)

        # Get bos_token_id
        if hasattr(tokenizer, 'bos_token_id') and tokenizer.bos_token_id:
            result["bos_token_id"] = int(tokenizer.bos_token_id)

        # Get pad_token_id (sometimes used as stop)
        if hasattr(tokenizer, 'pad_token_id') and tokenizer.pad_token_id:
            result["pad_token_id"] = int(tokenizer.pad_token_id)

        # Build stop_token_ids list
        stop_ids = []
        if result.get("eos_token_id"):
            stop_ids.append(result["eos_token_id"])
        if result.get("pad_token_id") and result["pad_token_id"] not in stop_ids:
            stop_ids.append(result["pad_token_id"])
        if stop_ids:
            result["stop_token_ids"] = stop_ids

        # Get vocab_size
        if hasattr(tokenizer, 'vocab_size') and tokenizer.vocab_size:
            result["vocab_size"] = int(tokenizer.vocab_size)
        elif hasattr(tokenizer, 'model_max_length'):
            result["vocab_size"] = int(tokenizer.model_max_length)

        return result
    except Exception as e:
        return {}


def get_special_tokens_for_model(template_type: str) -> dict:
    """
    Get special tokens for a known model family.

    Args:
        template_type: The detected template type (qwen, llama3, etc.)

    Returns:
        Dict with special token info, or empty dict if unknown
    """
    return MODEL_FAMILY_SPECIAL_TOKENS.get(template_type, {})


def get_special_tokens(config_json: Path = None,
                       tokenizer_json: Path = None,
                       model_name: str = None,
                       template_type: str = None) -> dict:
    """
    Get special token IDs from multiple sources.

    Priority:
    1. HuggingFace transformers (if model_name provided)
    2. Known model family defaults (if template_type provided)
    3. tokenizer.json (if provided)

    Returns dict with keys: eos_token_id, stop_token_ids, vocab_size
    """
    result = {}

    # Try HuggingFace transformers first
    if model_name:
        hf_tokens = get_special_tokens_from_hf(model_name)
        if hf_tokens:
            result.update(hf_tokens)
            return result

    # Try known model family defaults
    if template_type:
        family_tokens = get_special_tokens_for_model(template_type)
        if family_tokens:
            result.update(family_tokens)
            # Still try to get vocab_size from tokenizer.json if available
            if tokenizer_json and tokenizer_json.exists():
                vs = get_vocab_size(tokenizer_json)
                if vs > 0:
                    result["vocab_size"] = vs
            return result

    # Try tokenizer.json
    if tokenizer_json and tokenizer_json.exists():
        eos_id = find_eos_token_id(tokenizer_json)
        if eos_id > 0:
            result["eos_token_id"] = eos_id
            result["stop_token_ids"] = [eos_id]

        vs = get_vocab_size(tokenizer_json)
        if vs > 0:
            result["vocab_size"] = vs

    return result


# ═══════════════════════════════════════════════════════════════════════
# Template Application
# ═══════════════════════════════════════════════════════════════════════

def apply_template(template: str, prompt: str, system: str = None) -> str:
    """
    Apply chat template to user prompt.

    Args:
        template: Chat template string with {system} and {prompt} placeholders
        prompt: User message
        system: System prompt (or use default)

    Returns:
        Formatted prompt ready for tokenization
    """
    if system is None:
        system = ""

    result = template.replace("{system}", system)
    result = result.replace("{prompt}", prompt)
    return result


# ═══════════════════════════════════════════════════════════════════════
# Serialization for JSON/Manifest Files
# ═══════════════════════════════════════════════════════════════════════

def save_template_json(template_info: Dict[str, str], output_path: Path,
                       eos_token_id: int = -1,
                       stop_token_ids: list = None,
                       vocab_size: int = -1) -> None:
    """
    Save template info to JSON file.

    Output format:
    {
        "template": "...",
        "default_system": "...",
        "type": "qwen",
        "eos_token_id": 151643,
        "stop_token_ids": [151643, 151645],
        "vocab_size": 151936
    }
    """
    output = {
        "template": template_info["template"],
        "default_system": template_info.get("default_system", ""),
        "type": template_info.get("type", "none"),
        "eos_token_id": eos_token_id,
    }

    # Add stop_token_ids if provided
    if stop_token_ids:
        output["stop_token_ids"] = stop_token_ids

    # Add vocab_size if provided
    if vocab_size > 0:
        output["vocab_size"] = vocab_size

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"  Saved chat template to {output_path}")


def load_template_json(input_path: Path) -> Optional[Dict[str, Any]]:
    """
    Load template info from JSON file.

    Returns:
        Dict with 'template', 'default_system', 'type', 'eos_token_id'
    """
    if not input_path.exists():
        return None

    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"  Warning: Failed to load template from {input_path}: {e}")
        return None


def add_template_to_manifest(manifest: Dict, template_info: Dict[str, str],
                             eos_token_id: int = -1) -> Dict:
    """
    Add template info to weights manifest dictionary.

    Updates manifest dict in-place and returns it.
    """
    manifest["chat_template"] = {
        "type": template_info.get("type", "none"),
        "template_hash": hashlib.sha256(
            template_info["template"].encode('utf-8')
        ).hexdigest()[:16],
        "default_system": template_info.get("default_system", ""),
        "eos_token_id": eos_token_id,
    }
    return manifest


if __name__ == "__main__":
    # Simple test
    import sys

    if len(sys.argv) > 1:
        config_path = Path(sys.argv[1])
        tokenizer_json = Path(sys.argv[2]) if len(sys.argv) > 2 else None

        template = detect_template(config_path, tokenizer_json)
        print(json.dumps(template, indent=2))
    else:
        print("Usage: chat_template_utils.py <config.json> [tokenizer.json]")

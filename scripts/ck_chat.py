#!/usr/bin/env python3
"""
C-Kernel-Engine Chat Interface

Uses the HuggingFace tokenizer or GGUF tokenizer and calls the compiled C model library.
"""
from __future__ import annotations  # Python 3.9 compatibility

import argparse
import ctypes
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

# Try to import tokenizers (HuggingFace or our GGUF tokenizer)
HF_TOKENIZER_AVAILABLE = False
try:
    from tokenizers import Tokenizer
    HF_TOKENIZER_AVAILABLE = True
except ImportError:
    pass

# Always have GGUF tokenizer available
from gguf_tokenizer import GGUFTokenizer, Tokenizer as GGUFTokenizerWrapper


class CKModel:
    """Wrapper for the C model library."""

    def __init__(self, model_dir: str, parity_dir: str = None):
        self.model_dir = Path(model_dir)
        self.parity_dir = Path(parity_dir) if parity_dir else None
        self.lib = None
        self.tokenizer = None
        self.vocab_size = 0
        self.context_window = 0
        self.has_kv_decode = False
        self.has_parity = False

    def load(self, gguf_path: str = None) -> bool:
        """Load model library and tokenizer."""
        # Load tokenizer - try multiple sources
        tokenizer_json = self.model_dir / "tokenizer.json"
        vocab_json = self.model_dir / "vocab.json"

        if tokenizer_json.exists() and HF_TOKENIZER_AVAILABLE:
            # Use HuggingFace tokenizer if available
            self.tokenizer = Tokenizer.from_file(str(tokenizer_json))
            print(f"Loaded HuggingFace tokenizer from {tokenizer_json}")
        elif vocab_json.exists():
            # Use extracted vocab JSON
            self.tokenizer = GGUFTokenizerWrapper.from_file(str(vocab_json))
            print(f"Loaded GGUF tokenizer from {vocab_json}")
        elif gguf_path and Path(gguf_path).exists():
            # Extract directly from GGUF
            print(f"Extracting tokenizer from GGUF: {gguf_path}")
            self.tokenizer = GGUFTokenizerWrapper(GGUFTokenizer.from_gguf(gguf_path))
            # Save for next time
            self.tokenizer._tokenizer.save(str(vocab_json))
            print(f"Saved vocab to {vocab_json}")
        else:
            print(f"Error: No tokenizer found. Tried:")
            print(f"  - {tokenizer_json}")
            print(f"  - {vocab_json}")
            if gguf_path:
                print(f"  - {gguf_path}")
            return False

        # Load C library
        lib_path = self.model_dir / "libmodel.so"
        if not lib_path.exists():
            print(f"Error: Model library not found: {lib_path}")
            return False

        self.lib = ctypes.CDLL(str(lib_path))

        # Setup function signatures
        self.lib.ck_model_init.argtypes = [ctypes.c_char_p]
        self.lib.ck_model_init.restype = ctypes.c_int

        self.lib.ck_model_embed_tokens.argtypes = [ctypes.POINTER(ctypes.c_int32), ctypes.c_int]
        self.lib.ck_model_embed_tokens.restype = ctypes.c_int

        self.lib.ck_model_forward.argtypes = [ctypes.POINTER(ctypes.c_float)]
        self.lib.ck_model_forward.restype = ctypes.c_int

        self.lib.ck_model_get_logits.argtypes = []
        self.lib.ck_model_get_logits.restype = ctypes.POINTER(ctypes.c_float)

        self.lib.ck_model_get_vocab_size.argtypes = []
        self.lib.ck_model_get_vocab_size.restype = ctypes.c_int

        self.lib.ck_model_get_context_window.argtypes = []
        self.lib.ck_model_get_context_window.restype = ctypes.c_int

        self.lib.ck_model_get_active_tokens.argtypes = []
        self.lib.ck_model_get_active_tokens.restype = ctypes.c_int

        self.lib.ck_model_free.argtypes = []
        self.lib.ck_model_free.restype = None

        # Optional KV-cache decode API (newer generated runtimes).
        try:
            self.lib.ck_model_kv_cache_enable.argtypes = [ctypes.c_int]
            self.lib.ck_model_kv_cache_enable.restype = ctypes.c_int
            self.lib.ck_model_kv_cache_reset.argtypes = []
            self.lib.ck_model_kv_cache_reset.restype = None
            self.lib.ck_model_decode.argtypes = [ctypes.c_int32, ctypes.POINTER(ctypes.c_float)]
            self.lib.ck_model_decode.restype = ctypes.c_int
            self.has_kv_decode = True
        except AttributeError:
            self.has_kv_decode = False

        # Optional parity dump API (generated with --parity flag).
        try:
            self.lib.parity_set_output_dir.argtypes = [ctypes.c_char_p]
            self.lib.parity_set_output_dir.restype = None
            self.lib.parity_set_token_index.argtypes = [ctypes.c_int]
            self.lib.parity_set_token_index.restype = None
            self.has_parity = True
        except AttributeError:
            self.has_parity = False

        # Setup parity dir if requested
        if self.parity_dir and self.has_parity:
            self.parity_dir.mkdir(parents=True, exist_ok=True)
            # Keep reference to encoded string to prevent garbage collection
            self._parity_dir_bytes = str(self.parity_dir).encode()
            self.lib.parity_set_output_dir(self._parity_dir_bytes)

        # Initialize model
        weights_path = self.model_dir / "weights.bump"
        if not weights_path.exists():
            print(f"Error: Weights not found: {weights_path}")
            return False

        ret = self.lib.ck_model_init(str(weights_path).encode())
        if ret != 0:
            print(f"Error: Failed to initialize model (code {ret})")
            return False

        self.vocab_size = self.lib.ck_model_get_vocab_size()
        self.context_window = self.lib.ck_model_get_context_window()

        return True

    def kv_cache_enable(self, capacity: Optional[int] = None) -> bool:
        if not self.has_kv_decode:
            return False
        if capacity is None:
            capacity = self.context_window
        ret = self.lib.ck_model_kv_cache_enable(int(capacity))
        return ret == 0

    def kv_cache_reset(self):
        if self.has_kv_decode:
            self.lib.ck_model_kv_cache_reset()

    def set_parity_token_index(self, idx: int):
        """Set current token index for parity dumps."""
        if self.has_parity and self.parity_dir:
            self.lib.parity_set_token_index(idx)

    def encode(self, text: str) -> list:
        """Tokenize text."""
        return self.tokenizer.encode(text).ids

    def decode(self, token_ids: list) -> str:
        """Decode token IDs to text."""
        return self.tokenizer.decode(token_ids)

    def forward(self, token_ids: list) -> np.ndarray:
        """Run forward pass and return logits for last position."""
        n = len(token_ids)
        tokens = (ctypes.c_int32 * n)(*token_ids)

        self.lib.ck_model_embed_tokens(tokens, n)
        self.lib.ck_model_forward(None)

        # Get logits pointer
        logits_ptr = self.lib.ck_model_get_logits()
        active_tokens = self.lib.ck_model_get_active_tokens()

        # Get last position logits
        last_pos_offset = (active_tokens - 1) * self.vocab_size
        logits_array = np.ctypeslib.as_array(logits_ptr, shape=(active_tokens * self.vocab_size,))
        last_logits = logits_array[last_pos_offset:last_pos_offset + self.vocab_size].copy()

        return last_logits

    def prefill(self, token_ids: list) -> np.ndarray:
        """Prefill KV cache by running a full forward once (returns last logits)."""
        return self.forward(token_ids)

    def decode_step(self, token_id: int) -> np.ndarray:
        """Decode one token using KV cache (returns logits for that token)."""
        ret = self.lib.ck_model_decode(ctypes.c_int32(int(token_id)), None)
        if ret != 0:
            raise RuntimeError(f"ck_model_decode failed (code {ret})")

        logits_ptr = self.lib.ck_model_get_logits()
        active_tokens = self.lib.ck_model_get_active_tokens()
        last_pos_offset = (active_tokens - 1) * self.vocab_size
        logits_array = np.ctypeslib.as_array(logits_ptr, shape=(active_tokens * self.vocab_size,))
        return logits_array[last_pos_offset:last_pos_offset + self.vocab_size].copy()

    def free(self):
        """Free model resources."""
        if self.lib:
            self.lib.ck_model_free()


def sample_top_k(logits: np.ndarray, k: int = 40, temperature: float = 0.7) -> int:
    """Sample from top-k logits with temperature."""
    if temperature <= 0:
        return int(np.argmax(logits))

    # Apply temperature
    logits = logits / temperature

    # Top-k filtering
    top_k_indices = np.argpartition(logits, -k)[-k:]
    top_k_logits = logits[top_k_indices]

    # Softmax
    max_logit = np.max(top_k_logits)
    exp_logits = np.exp(top_k_logits - max_logit)
    probs = exp_logits / np.sum(exp_logits)

    # Sample
    idx = np.random.choice(len(top_k_indices), p=probs)
    return int(top_k_indices[idx])


def generate(model: CKModel, prompt: str, max_tokens: int = 50,
             temperature: float = 0.7, verbose: bool = False,
             show_stats: bool = True) -> str:
    """Generate text from prompt."""
    # Tokenize prompt
    token_ids = model.encode(prompt)
    prompt_tokens = len(token_ids)

    if verbose:
        print(f"[Prompt tokens: {prompt_tokens}]")

    generated = []
    sample_times = []
    decode_times = []
    prefill_time = 0.0
    start_time = time.time()

    if model.has_kv_decode and model.kv_cache_enable():
        # KV-cache path: prefill once, then decode token-by-token.
        t0 = time.time()
        model.set_parity_token_index(0)  # Token 0 for prefill output
        logits = model.prefill(token_ids)
        prefill_time = time.time() - t0

        # NaN detection
        if np.isnan(logits).any():
            nan_count = np.isnan(logits).sum()
            print(f"\n[DEBUG] NaN in prefill logits: {nan_count}/{logits.size} values")
            print(f"[DEBUG] logits shape: {logits.shape}, dtype: {logits.dtype}")
            print(f"[DEBUG] logits range: [{np.nanmin(logits):.3e}, {np.nanmax(logits):.3e}]")

        for i in range(max_tokens):
            # Sample
            t_sample = time.time()
            next_token = sample_top_k(logits, k=40, temperature=temperature)
            sample_times.append(time.time() - t_sample)

            if next_token <= 2:
                break
            generated.append(next_token)
            token_ids.append(next_token)
            token_text = model.decode([next_token])
            print(token_text, end='', flush=True)
            if len(token_ids) >= model.context_window - 1:
                break

            # Set parity token index before decode
            model.set_parity_token_index(prompt_tokens + i)

            # Decode step
            t_decode = time.time()
            logits = model.decode_step(next_token)
            decode_times.append(time.time() - t_decode)

            # NaN detection
            if np.isnan(logits).any():
                nan_count = np.isnan(logits).sum()
                print(f"\n[DEBUG] NaN in decode logits (step {i}): {nan_count}/{logits.size} values")
    else:
        for i in range(max_tokens):
            # Forward pass (first is prefill, rest are decode)
            t0 = time.time()
            logits = model.forward(token_ids)
            fwd_time = time.time() - t0

            if i == 0:
                prefill_time = fwd_time
            else:
                decode_times.append(fwd_time)

            # Sample next token
            t_sample = time.time()
            next_token = sample_top_k(logits, k=40, temperature=temperature)
            sample_times.append(time.time() - t_sample)

            # Check for EOS (typically 0, 1, or 2)
            if next_token <= 2:
                break

            generated.append(next_token)
            token_ids.append(next_token)

            # Decode and print incrementally
            token_text = model.decode([next_token])
            print(token_text, end='', flush=True)

            # Check context limit
            if len(token_ids) >= model.context_window - 1:
                break

    total_time = time.time() - start_time
    gen_count = len(generated)

    # Print statistics (llama.cpp style)
    if show_stats and gen_count > 0:
        print()  # newline after generated text

        # Prefill stats
        prefill_ms = prefill_time * 1000
        prefill_ms_per_token = prefill_ms / prompt_tokens if prompt_tokens > 0 else 0
        prefill_tps = prompt_tokens / prefill_time if prefill_time > 0 else 0

        # Decode stats
        total_decode_time = sum(decode_times) if decode_times else 0
        decode_ms = total_decode_time * 1000
        decode_count = len(decode_times)
        decode_ms_per_token = decode_ms / decode_count if decode_count > 0 else 0
        decode_tps = decode_count / total_decode_time if total_decode_time > 0 else 0

        # Sample stats
        total_sample_time = sum(sample_times) if sample_times else 0
        sample_ms = total_sample_time * 1000
        sample_count = len(sample_times)
        sample_ms_per_token = sample_ms / sample_count if sample_count > 0 else 0

        # Total stats
        total_ms = total_time * 1000
        total_tokens = prompt_tokens + gen_count

        print(f"\n\033[90m" +
              f"prompt eval: {prefill_ms:8.2f} ms / {prompt_tokens:4d} tokens ({prefill_ms_per_token:7.2f} ms/tok, {prefill_tps:7.2f} tok/s)\n" +
              f"      decode: {decode_ms:8.2f} ms / {decode_count:4d} runs   ({decode_ms_per_token:7.2f} ms/tok, {decode_tps:7.2f} tok/s)\n" +
              f"      sample: {sample_ms:8.2f} ms / {sample_count:4d} runs   ({sample_ms_per_token:7.2f} ms/tok)\n" +
              f"       total: {total_ms:8.2f} ms / {total_tokens:4d} tokens\033[0m")

    elif verbose:
        tokens_per_sec = gen_count / total_time if total_time > 0 else 0
        print(f"\n[Generated {gen_count} tokens in {total_time:.2f}s ({tokens_per_sec:.1f} tok/s)]")

    return model.decode(generated)


def chat_loop(model: CKModel, temperature: float = 0.7, max_tokens: int = 100,
              show_stats: bool = True):
    """Interactive chat loop."""
    print("\n" + "=" * 60)
    print("  C-Kernel-Engine Chat")
    print("  Type your message and press Enter. Commands: /exit, /help, /stats")
    print("=" * 60 + "\n")

    conversation = []

    while True:
        try:
            user_input = input("\033[92mYou: \033[0m").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ['/exit', '/quit', 'exit', 'quit']:
            print("Goodbye!")
            break

        if user_input.lower() == '/help':
            print("  Commands:")
            print("    /exit, /quit  - Exit the chat")
            print("    /stats        - Toggle performance stats display")
            print("    /help         - Show this help")
            continue

        if user_input.lower() == '/stats':
            show_stats = not show_stats
            print(f"  Performance stats: {'ON' if show_stats else 'OFF'}")
            continue

        # Build prompt (simple format)
        prompt = user_input

        # Generate response
        print("\033[94mAssistant: \033[0m", end='', flush=True)
        response = generate(model, prompt, max_tokens=max_tokens,
                          temperature=temperature, verbose=False,
                          show_stats=show_stats)
        print()


def main():
    parser = argparse.ArgumentParser(description="C-Kernel-Engine Chat Interface")
    parser.add_argument("--model-dir", required=True, help="Path to model directory")
    parser.add_argument("--gguf", help="Path to GGUF file (for tokenizer extraction)")
    parser.add_argument("--prompt", help="Single prompt (non-interactive mode)")
    parser.add_argument("--max-tokens", type=int, default=100, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--stats", action="store_true", default=True,
                       help="Show performance stats (default: on)")
    parser.add_argument("--no-stats", action="store_false", dest="stats",
                       help="Disable performance stats")
    parser.add_argument("--parity", action="store_true",
                       help="Save intermediate buffers for comparison (requires --parity in codegen)")
    parser.add_argument("--parity-dir",
                       help="Directory for parity dumps (default: model-dir/parity)")
    args = parser.parse_args()

    # Determine parity directory
    parity_dir = None
    if args.parity:
        parity_dir = args.parity_dir or str(Path(args.model_dir) / "parity")
        print(f"Parity dumps will be saved to: {parity_dir}")

    # Load model
    print(f"Loading model from {args.model_dir}...")
    model = CKModel(args.model_dir, parity_dir=parity_dir)

    if not model.load(gguf_path=args.gguf):
        sys.exit(1)

    print(f"Model loaded! Vocab: {model.vocab_size}, Context: {model.context_window}")

    try:
        if args.prompt:
            # Single prompt mode
            print(f"\nPrompt: {args.prompt}")
            print("Response: ", end='', flush=True)
            generate(model, args.prompt, max_tokens=args.max_tokens,
                    temperature=args.temperature, verbose=args.verbose,
                    show_stats=args.stats)
            print()
        else:
            # Interactive chat mode
            chat_loop(model, temperature=args.temperature, max_tokens=args.max_tokens,
                     show_stats=args.stats)
    finally:
        model.free()


if __name__ == "__main__":
    main()

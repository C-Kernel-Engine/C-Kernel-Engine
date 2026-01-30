#!/usr/bin/env python3
"""
Regression tests for v6.6 decode functionality.

These tests ensure the decode phase works correctly after prefill.
The key bug that was fixed: logits were written to position 0 but Python
read from position (active_tokens - 1), causing garbage output after first decode.

Run with: pytest tests/test_v66_decode_regression.py -v
"""

import ctypes
import numpy as np
import os
import pytest
import subprocess
import sys
from pathlib import Path

# Model directory
V66_DIR = Path(os.path.expanduser("~/.cache/ck-engine-v6.6/models/Qwen--Qwen2-0.5B-Instruct-GGUF"))
V65_DIR = Path(os.path.expanduser("~/.cache/ck-engine-v6.5/models/Qwen--Qwen2-0.5B-Instruct-GGUF"))

# Test configuration
VOCAB_SIZE = 151936
MAX_DECODE_TOKENS = 10


def load_v66_model():
    """Load v6.6 model library."""
    so_path = V66_DIR / "libmodel.so"
    if not so_path.exists():
        so_path = V66_DIR / "ck-kernel-inference.so"
    if not so_path.exists():
        pytest.skip(f"v6.6 model not found at {V66_DIR}")

    lib = ctypes.CDLL(str(so_path))

    # Setup function signatures
    lib.ck_model_init.argtypes = [ctypes.c_char_p]
    lib.ck_model_init.restype = ctypes.c_int

    lib.ck_model_embed_tokens.argtypes = [ctypes.POINTER(ctypes.c_int32), ctypes.c_int]
    lib.ck_model_embed_tokens.restype = ctypes.c_int

    lib.ck_model_forward.argtypes = [ctypes.POINTER(ctypes.c_float)]
    lib.ck_model_forward.restype = ctypes.c_int

    lib.ck_model_decode.argtypes = [ctypes.c_int32, ctypes.POINTER(ctypes.c_float)]
    lib.ck_model_decode.restype = ctypes.c_int

    lib.ck_model_get_logits.argtypes = []
    lib.ck_model_get_logits.restype = ctypes.POINTER(ctypes.c_float)

    lib.ck_model_get_active_tokens.argtypes = []
    lib.ck_model_get_active_tokens.restype = ctypes.c_int

    lib.ck_model_get_vocab_size.argtypes = []
    lib.ck_model_get_vocab_size.restype = ctypes.c_int

    lib.ck_model_kv_cache_enable.argtypes = [ctypes.c_int]
    lib.ck_model_kv_cache_enable.restype = ctypes.c_int

    lib.ck_model_kv_cache_reset.argtypes = []
    lib.ck_model_kv_cache_reset.restype = None

    lib.ck_model_free.argtypes = []
    lib.ck_model_free.restype = None

    # Initialize
    weights_path = V66_DIR / "weights.bump"
    ret = lib.ck_model_init(str(weights_path).encode())
    if ret != 0:
        pytest.fail(f"Model init failed with code {ret}")

    return lib


def get_logits(lib, position: int) -> np.ndarray:
    """Get logits at specified position."""
    vocab_size = lib.ck_model_get_vocab_size()
    logits_ptr = lib.ck_model_get_logits()
    offset = position * vocab_size
    logits_array = np.ctypeslib.as_array(logits_ptr, shape=((position + 1) * vocab_size,))
    return logits_array[offset:offset + vocab_size].copy()


def sample_argmax(logits: np.ndarray) -> int:
    """Simple argmax sampling."""
    return int(np.argmax(logits))


class TestV66DecodeRegression:
    """Test suite for v6.6 decode regression."""

    @pytest.fixture(autouse=True)
    def setup_teardown(self):
        """Setup and teardown for each test."""
        self.lib = load_v66_model()
        yield
        self.lib.ck_model_free()

    def test_prefill_produces_valid_logits(self):
        """Test that prefill produces valid (non-NaN, non-garbage) logits."""
        # Enable KV cache
        self.lib.ck_model_kv_cache_enable(100)
        self.lib.ck_model_kv_cache_reset()

        # Prefill with simple tokens (Hello! template)
        tokens = [151644, 872, 198, 9707, 0, 151645, 198, 151644, 77091, 198]
        tokens_arr = (ctypes.c_int32 * len(tokens))(*tokens)

        ret = self.lib.ck_model_embed_tokens(tokens_arr, len(tokens))
        assert ret == 0, f"embed_tokens failed with {ret}"

        self.lib.ck_model_forward(None)

        # Get logits for last position
        active_tokens = self.lib.ck_model_get_active_tokens()
        assert active_tokens == len(tokens), f"Expected {len(tokens)} tokens, got {active_tokens}"

        logits = get_logits(self.lib, active_tokens - 1)

        # Check for NaN/Inf
        assert not np.any(np.isnan(logits)), "Prefill produced NaN logits"
        assert not np.any(np.isinf(logits)), "Prefill produced Inf logits"

        # Check logits are in reasonable range
        assert np.max(logits) < 100, f"Logits max too high: {np.max(logits)}"
        assert np.min(logits) > -100, f"Logits min too low: {np.min(logits)}"

    def test_first_decode_produces_valid_token(self):
        """Test that first decode after prefill produces a valid token."""
        self.lib.ck_model_kv_cache_enable(100)
        self.lib.ck_model_kv_cache_reset()

        # Prefill
        tokens = [151644, 872, 198, 9707, 0, 151645, 198, 151644, 77091, 198]
        tokens_arr = (ctypes.c_int32 * len(tokens))(*tokens)
        self.lib.ck_model_embed_tokens(tokens_arr, len(tokens))
        self.lib.ck_model_forward(None)

        # Get first token from prefill
        active_tokens = self.lib.ck_model_get_active_tokens()
        logits = get_logits(self.lib, active_tokens - 1)
        first_token = sample_argmax(logits)

        # First decode step
        ret = self.lib.ck_model_decode(ctypes.c_int32(first_token), None)
        assert ret == 0, f"decode failed with {ret}"

        # Get decode logits
        active_tokens = self.lib.ck_model_get_active_tokens()
        decode_logits = get_logits(self.lib, active_tokens - 1)

        # Check decode logits are valid
        assert not np.any(np.isnan(decode_logits)), "First decode produced NaN logits"
        assert not np.any(np.isinf(decode_logits)), "First decode produced Inf logits"

        second_token = sample_argmax(decode_logits)
        assert 0 <= second_token < VOCAB_SIZE, f"Invalid token: {second_token}"

    def test_multiple_decode_steps_produce_valid_tokens(self):
        """Test that multiple decode steps produce valid tokens (the main regression test)."""
        self.lib.ck_model_kv_cache_enable(100)
        self.lib.ck_model_kv_cache_reset()

        # Prefill
        tokens = [151644, 872, 198, 9707, 0, 151645, 198, 151644, 77091, 198]
        tokens_arr = (ctypes.c_int32 * len(tokens))(*tokens)
        self.lib.ck_model_embed_tokens(tokens_arr, len(tokens))
        self.lib.ck_model_forward(None)

        active_tokens = self.lib.ck_model_get_active_tokens()
        logits = get_logits(self.lib, active_tokens - 1)

        generated_tokens = []
        for step in range(MAX_DECODE_TOKENS):
            next_token = sample_argmax(logits)
            generated_tokens.append(next_token)

            # Decode step
            ret = self.lib.ck_model_decode(ctypes.c_int32(next_token), None)
            assert ret == 0, f"decode step {step} failed with {ret}"

            # Get new logits
            active_tokens = self.lib.ck_model_get_active_tokens()
            logits = get_logits(self.lib, active_tokens - 1)

            # Verify logits are valid
            assert not np.any(np.isnan(logits)), f"Decode step {step} produced NaN logits"
            assert not np.any(np.isinf(logits)), f"Decode step {step} produced Inf logits"

        # All tokens should be valid vocab indices
        for i, tok in enumerate(generated_tokens):
            assert 0 <= tok < VOCAB_SIZE, f"Step {i} produced invalid token: {tok}"

    def test_decode_logits_are_at_correct_position(self):
        """Test that decode writes logits to the position-indexed location."""
        self.lib.ck_model_kv_cache_enable(100)
        self.lib.ck_model_kv_cache_reset()

        # Prefill
        tokens = [151644, 872, 198, 9707, 0, 151645, 198, 151644, 77091, 198]
        num_prefill = len(tokens)
        tokens_arr = (ctypes.c_int32 * num_prefill)(*tokens)
        self.lib.ck_model_embed_tokens(tokens_arr, num_prefill)
        self.lib.ck_model_forward(None)

        logits = get_logits(self.lib, num_prefill - 1)
        first_token = sample_argmax(logits)

        # First decode
        self.lib.ck_model_decode(ctypes.c_int32(first_token), None)
        active = self.lib.ck_model_get_active_tokens()
        assert active == num_prefill + 1, f"Expected {num_prefill + 1} tokens, got {active}"

        # Logits at position num_prefill should be valid
        decode_logits = get_logits(self.lib, num_prefill)
        assert not np.any(np.isnan(decode_logits)), "Decode logits at correct position have NaN"

        # Second decode
        second_token = sample_argmax(decode_logits)
        self.lib.ck_model_decode(ctypes.c_int32(second_token), None)
        active = self.lib.ck_model_get_active_tokens()
        assert active == num_prefill + 2, f"Expected {num_prefill + 2} tokens, got {active}"

        # Logits at position num_prefill + 1 should be valid
        decode_logits_2 = get_logits(self.lib, num_prefill + 1)
        assert not np.any(np.isnan(decode_logits_2)), "Second decode logits have NaN"


class TestV66VsV65Parity:
    """Test v6.6 output quality matches v6.5."""

    def run_inference(self, version: str, prompt: str, max_tokens: int = 5) -> str:
        """Run inference with specified version."""
        if version == "v6.5":
            cmd = [
                sys.executable, "scripts/v6.5/ck_run_v6_5.py", "run",
                "Qwen/Qwen2-0.5B-Instruct-GGUF",
                "--weight-dtype=q4_k_m",
                f"--prompt={prompt}",
                f"--max-tokens={max_tokens}"
            ]
        else:
            cmd = [
                sys.executable, "version/v6.6/scripts/ck_run_v6_6.py", "run",
                "Qwen/Qwen2-0.5B-Instruct-GGUF",
                "--weight-dtype=q4_k_m",
                "--context-len=100",
                f"--prompt={prompt}",
                f"--max-tokens={max_tokens}"
            ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

        # Extract response
        for line in result.stdout.split('\n'):
            if 'Response:' in line:
                return line.split('Response:')[1].strip()
        return None

    def test_both_versions_produce_coherent_output(self):
        """Test that both v6.5 and v6.6 produce coherent (non-gibberish) output."""
        prompt = "What is 2+2?"

        v65_response = self.run_inference("v6.5", prompt, max_tokens=10)
        v66_response = self.run_inference("v6.6", prompt, max_tokens=10)

        assert v65_response is not None, "v6.5 produced no response"
        assert v66_response is not None, "v6.6 produced no response"

        # Both should mention "4" somewhere
        assert "4" in v65_response or "four" in v65_response.lower(), \
            f"v6.5 didn't answer correctly: {v65_response}"
        assert "4" in v66_response or "four" in v66_response.lower(), \
            f"v6.6 didn't answer correctly: {v66_response}"

    def test_v66_produces_english_words(self):
        """Test that v6.6 produces recognizable English words (not gibberish)."""
        prompt = "Hello!"
        response = self.run_inference("v6.6", prompt, max_tokens=10)

        assert response is not None, "v6.6 produced no response"

        # Response should contain common English words
        words = response.lower().split()
        common_words = {'i', 'the', 'a', 'to', 'is', 'am', 'are', 'you', 'hello', 'hi',
                       'how', 'can', 'help', 'what', 'my', 'name', 'your'}

        found_common = any(word.strip('.,!?') in common_words for word in words)
        assert found_common, f"v6.6 response doesn't contain common English words: {response}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

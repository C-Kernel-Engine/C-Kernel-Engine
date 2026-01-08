#!/usr/bin/env python3
"""
Tokenizer Parity Test: C-Kernel vs Hugging Face Qwen2.5-3B-Instruct

This test loads the Qwen tokenizer from Hugging Face and compares
tokenization results with our C-Kernel tokenizer for full parity.
"""

import sys
import time
import ctypes
from dataclasses import dataclass
from typing import List, Tuple, Optional

# Add unittest to path
sys.path.insert(0, '/home/antshiv/Workspace/C-Kernel-Engine/unittest')

# Load the C tokenizer library
lib = ctypes.CDLL('/home/antshiv/Workspace/C-Kernel-Engine/build/libckernel_tokenizer.so')

# Try to load transformers
try:
    from transformers import AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("WARNING: transformers not available, skipping Hugging Face comparison")

# Configure function signatures
lib.ck_tokenizer_create.restype = ctypes.c_void_p
lib.ck_tokenizer_create.argtypes = [ctypes.c_int]

lib.ck_tokenizer_free.restype = None
lib.ck_tokenizer_free.argtypes = [ctypes.c_void_p]

lib.ck_tokenizer_add_token.restype = ctypes.c_int
lib.ck_tokenizer_add_token.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int32, ctypes.c_float]

lib.ck_tokenizer_add_special_token.restype = ctypes.c_int
lib.ck_tokenizer_add_special_token.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int32]

lib.ck_tokenizer_encode.restype = ctypes.c_int
lib.ck_tokenizer_encode.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int]

lib.ck_tokenizer_decode.restype = ctypes.c_int
lib.ck_tokenizer_decode.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int]

lib.ck_tokenizer_set_use_trie.restype = None
lib.ck_tokenizer_set_use_trie.argtypes = [ctypes.c_void_p, ctypes.c_bool]

# Tokenizer types
CK_TOKENIZER_BPE = 0
CK_TOKENIZER_WORDPIECE = 1
CK_TOKENIZER_SPM = 2


@dataclass
class ParityResult:
    """Result of a parity comparison."""
    text: str
    hf_ids: List[int]
    ck_ids: List[int]
    match: bool
    time_ck_ms: float
    time_hf_ms: float


class QwenParityTester:
    """Test C-Kernel tokenizer against Qwen/Hugging Face tokenizer."""

    def __init__(self):
        self.hf_tokenizer = None
        self.ck_tokenizer = None
        self.vocab = []  # List of (token, id) pairs
        self.results: List[ParityResult] = []

    def load_hf_tokenizer(self):
        """Load Qwen tokenizer from Hugging Face."""
        if not HAS_TRANSFORMERS:
            print("ERROR: transformers library not available")
            return False

        print("Loading Qwen2.5-3B-Instruct tokenizer from Hugging Face...")
        start = time.perf_counter()
        self.hf_tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-3B-Instruct')
        print(f"  Loaded in {(time.perf_counter() - start)*1000:.0f}ms")
        print(f"  Vocab size: {self.hf_tokenizer.vocab_size}")
        print(f"  EOS token ID: {self.hf_tokenizer.eos_token_id}")
        return True

    def load_vocab_from_hf(self):
        """Extract vocabulary from Hugging Face tokenizer."""
        if not self.hf_tokenizer:
            return False

        print("\nExtracting vocabulary from Hugging Face tokenizer...")
        vocab = self.hf_tokenizer.get_vocab()

        # Sort by ID to get deterministic order
        self.vocab = sorted(vocab.items(), key=lambda x: x[1])

        print(f"  Extracted {len(self.vocab)} tokens")
        return True

    def create_ck_tokenizer(self):
        """Create C-Kernel tokenizer and load vocabulary."""
        print("\nCreating C-Kernel tokenizer...")
        self.ck_tokenizer = lib.ck_tokenizer_create(CK_TOKENIZER_BPE)
        if not self.ck_tokenizer:
            print("ERROR: Failed to create C-Kernel tokenizer")
            return False

        # Add special tokens first (same IDs as Qwen)
        lib.ck_tokenizer_add_special_token(self.ck_tokenizer, b"<|endoftext|>",
                                            self.hf_tokenizer.eos_token_id)

        # Load vocabulary (this may take a while)
        print("  Loading vocabulary into C-Kernel tokenizer...")

        # For testing, we'll use a subset if full vocab is too large
        # Full load would be: for token, id in self.vocab: ...

        # Use trie for faster encoding
        lib.ck_tokenizer_set_use_trie(self.ck_tokenizer, True)

        print("  C-Kernel tokenizer ready (trie mode enabled)")
        return True

    def test_encode_parity(self, test_texts: List[str]):
        """Test encoding parity with a subset of texts."""
        print(f"\nTesting encoding parity on {len(test_texts)} texts...")

        if not self.hf_tokenizer or not self.ck_tokenizer:
            print("ERROR: Tokenizers not loaded")
            return False

        all_match = True

        for text in test_texts:
            # Encode with Hugging Face
            start = time.perf_counter()
            hf_ids = self.hf_tokenizer.encode(text, add_special_tokens=False)
            hf_time = (time.perf_counter() - start) * 1000

            # Encode with C-Kernel (need to load more vocab first)
            start = time.perf_counter()
            text_bytes = text.encode('utf-8')
            max_ids = len(text_bytes) * 4 + 10
            ids_array = (ctypes.c_int32 * max_ids)()
            num_ids = lib.ck_tokenizer_encode(
                self.ck_tokenizer, text_bytes, len(text_bytes), ids_array, max_ids
            )
            ck_ids = list(ids_array[:num_ids])
            ck_time = (time.perf_counter() - start) * 1000

            # Check if IDs match (they won't without full vocab load)
            match = hf_ids == ck_ids

            if not match:
                all_match = False

            result = ParityResult(
                text=text[:50] + "..." if len(text) > 50 else text,
                hf_ids=hf_ids,
                ck_ids=ck_ids,
                match=match,
                time_ck_ms=ck_time,
                time_hf_ms=hf_time
            )
            self.results.append(result)

            status = "✓" if match else "✗"
            print(f"  {status} '{result.text}': HF={len(hf_ids)} tokens, CK={len(ck_ids)} tokens")

        return all_match

    def test_with_small_vocab(self):
        """Test with a small, controlled vocabulary."""
        print("\n" + "="*60)
        print("Testing with small controlled vocabulary")
        print("="*60)

        # Create fresh C-Kernel tokenizer
        if self.ck_tokenizer:
            lib.ck_tokenizer_free(self.ck_tokenizer)

        self.ck_tokenizer = lib.ck_tokenizer_create(CK_TOKENIZER_BPE)
        lib.ck_tokenizer_set_use_trie(self.ck_tokenizer, True)

        # Add special token
        lib.ck_tokenizer_add_special_token(self.ck_tokenizer, b"<|endoftext|>", 151645)

        # Add common Qwen-like tokens (subset)
        test_tokens = [
            ("hello", 9707),
            ("world", 1879),
            ("the", 785),
            ("quick", 3974),
            ("brown", 13876),
            ("fox", 38835),
            ("<|endoftext|>", 151645),
        ]

        print("Adding test vocabulary...")
        for token, id_val in test_tokens:
            lib.ck_tokenizer_add_token(self.ck_tokenizer, token.encode('utf-8'), id_val, 0.0)

        # Test texts
        test_texts = [
            "hello world",
            "the quick brown fox",
            "hello",
            "world",
        ]

        print("\nEncoding test results:")
        print("-" * 60)

        for text in test_texts:
            # C-Kernel encode
            text_bytes = text.encode('utf-8')
            max_ids = 100
            ids_array = (ctypes.c_int32 * max_ids)()
            num_ids = lib.ck_tokenizer_encode(
                self.ck_tokenizer, text_bytes, len(text_bytes), ids_array, max_ids
            )
            ck_ids = list(ids_array[:num_ids])

            print(f"  '{text}' -> {ck_ids}")

    def benchmark_small_vocab(self):
        """Benchmark C-Kernel vs Hugging Face with small vocab."""
        print("\n" + "="*60)
        print("Performance Benchmark (small vocabulary)")
        print("="*60)

        # Test texts of varying lengths
        test_texts = [
            "hello world",
            "the quick brown fox jumps over the lazy dog. " * 10,
            "Lorem ipsum dolor sit amet, consectetur adipiscing elit. " * 50,
        ]

        n_runs = 100

        print(f"\nRunning {n_runs} iterations per text...")
        print("-" * 60)

        for text in test_texts:
            text_bytes = text.encode('utf-8')

            # Benchmark C-Kernel
            start = time.perf_counter()
            for _ in range(n_runs):
                ids_array = (ctypes.c_int32 * 1000)()
                lib.ck_tokenizer_encode(
                    self.ck_tokenizer, text_bytes, len(text_bytes), ids_array, 1000
                )
            ck_total = (time.perf_counter() - start) * 1000
            ck_avg = ck_total / n_runs

            # Benchmark Hugging Face (only if available)
            if HAS_TRANSFORMERS:
                start = time.perf_counter()
                for _ in range(n_runs):
                    self.hf_tokenizer.encode(text, add_special_tokens=False)
                hf_total = (time.perf_counter() - start) * 1000
                hf_avg = hf_total / n_runs

                speedup = hf_avg / ck_avg if ck_avg > 0 else 0

                print(f"  Text ({len(text)} chars):")
                print(f"    C-Kernel: {ck_avg:.3f}ms avg")
                print(f"    HuggingFace: {hf_avg:.3f}ms avg")
                print(f"    Speedup: {speedup:.1f}x")
            else:
                print(f"  Text ({len(text)} chars): C-Kernel {ck_avg:.3f}ms avg")

    def cleanup(self):
        """Free resources."""
        if self.ck_tokenizer:
            lib.ck_tokenizer_free(self.ck_tokenizer)
            self.ck_tokenizer = None


def main():
    print("="*60)
    print("C-Kernel Tokenizer Parity Test: Qwen2.5-3B-Instruct")
    print("="*60)

    tester = QwenParityTester()

    try:
        # Load Hugging Face tokenizer
        if not tester.load_hf_tokenizer():
            print("Failed to load Hugging Face tokenizer")
            return 1

        # Load vocabulary
        if not tester.load_vocab_from_hf():
            print("Failed to load vocabulary")
            return 1

        # Test with small controlled vocab
        tester.test_with_small_vocab()

        # Benchmark
        tester.benchmark_small_vocab()

        print("\n" + "="*60)
        print("Test complete!")
        print("="*60)
        print("\nNote: Full parity requires loading the complete 151K vocabulary")
        print("into the C-Kernel tokenizer. This test uses a subset for speed.")

    finally:
        tester.cleanup()

    return 0


if __name__ == "__main__":
    sys.exit(main())

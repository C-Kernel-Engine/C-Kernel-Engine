#!/usr/bin/env python3
"""
Full Tokenizer Parity Test: C-Kernel vs HuggingFace

This test loads the FULL vocabulary from HuggingFace into the C tokenizer
and verifies exact parity on encoding.

Parity Status:
- ASCII text: FULL PARITY (spaces converted to Ġ correctly)
- UTF-8 text: PARTIAL (requires byte-level BPE encoding - not yet implemented)
  - GPT-2/Qwen use byte-level BPE where UTF-8 bytes are mapped to special chars
  - Chinese, Russian, emojis need byte encoding table to work

The foundation tests (test_tokenizer.py) test core logic with custom vocabulary.
This test verifies real-world GGUF/HuggingFace compatibility.

Usage:
    python unittest/test_tokenizer_full_parity.py [--model MODEL] [--ascii-only]
"""

import sys
import time
import ctypes
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "unittest"))

# Load the C tokenizer library
try:
    lib = ctypes.CDLL(str(ROOT / "build" / "libckernel_tokenizer.so"))
except Exception as e:
    print(f"ERROR: Could not load tokenizer library: {e}")
    print("Run 'make libckernel_tokenizer.so' first")
    sys.exit(1)

# Configure function signatures
lib.ck_tokenizer_create.restype = ctypes.c_void_p
lib.ck_tokenizer_create.argtypes = [ctypes.c_int]

lib.ck_tokenizer_free.restype = None
lib.ck_tokenizer_free.argtypes = [ctypes.c_void_p]

lib.ck_tokenizer_add_token.restype = ctypes.c_int
lib.ck_tokenizer_add_token.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int32, ctypes.c_float]

lib.ck_tokenizer_encode.restype = ctypes.c_int
lib.ck_tokenizer_encode.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int]

lib.ck_tokenizer_set_use_trie.restype = None
lib.ck_tokenizer_set_use_trie.argtypes = [ctypes.c_void_p, ctypes.c_bool]

lib.ck_tokenizer_set_special_ids.restype = None
lib.ck_tokenizer_set_special_ids.argtypes = [ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32,
                                              ctypes.c_int32, ctypes.c_int32, ctypes.c_int32]

lib.ck_tokenizer_set_space_prefix_style.restype = None
lib.ck_tokenizer_set_space_prefix_style.argtypes = [ctypes.c_void_p, ctypes.c_int]

lib.ck_tokenizer_detect_space_prefix_style.restype = ctypes.c_int
lib.ck_tokenizer_detect_space_prefix_style.argtypes = [ctypes.c_void_p]

CK_TOKENIZER_BPE = 0
CK_SPACE_PREFIX_AUTO = 0
CK_SPACE_PREFIX_GPT2 = 1
CK_SPACE_PREFIX_SPM = 2


def load_hf_vocab_into_c_tokenizer(ck_tok, hf_tokenizer):
    """Load vocabulary from HuggingFace tokenizer into C tokenizer."""
    vocab = hf_tokenizer.get_vocab()
    print(f"  Loading {len(vocab)} tokens from HuggingFace...")

    start = time.perf_counter()
    loaded = 0
    failed = 0

    for token_str, token_id in vocab.items():
        try:
            # Encode token to bytes (handle special characters)
            token_bytes = token_str.encode('utf-8')
            ret = lib.ck_tokenizer_add_token(ck_tok, token_bytes, token_id, 0.0)
            if ret == 0:
                loaded += 1
            else:
                failed += 1
        except Exception as e:
            failed += 1
            if failed < 10:
                print(f"    Warning: Could not add token {repr(token_str)}: {e}")

    elapsed = time.perf_counter() - start
    print(f"  Loaded {loaded} tokens in {elapsed:.2f}s ({loaded/elapsed:.0f} tokens/sec)")
    if failed > 0:
        print(f"  Warning: {failed} tokens failed to load")

    return loaded


def compare_encoding(ck_tok, hf_tok, text: str, show_details: bool = True) -> bool:
    """Compare encoding between C and HuggingFace tokenizers."""
    text_bytes = text.encode('utf-8')
    max_ids = len(text_bytes) * 4 + 100

    # HuggingFace encode
    hf_start = time.perf_counter()
    hf_ids = hf_tok.encode(text, add_special_tokens=False)
    hf_time = (time.perf_counter() - hf_start) * 1000

    # C-Kernel encode
    ids_array = (ctypes.c_int32 * max_ids)()
    ck_start = time.perf_counter()
    num_ids = lib.ck_tokenizer_encode(ck_tok, text_bytes, len(text_bytes), ids_array, max_ids)
    ck_time = (time.perf_counter() - ck_start) * 1000
    ck_ids = list(ids_array[:num_ids])

    match = hf_ids == ck_ids
    speedup = hf_time / ck_time if ck_time > 0 else float('inf')

    if show_details:
        status = "PASS" if match else "FAIL"
        text_preview = text[:50] + "..." if len(text) > 50 else text
        print(f"  [{status}] \"{text_preview}\"")
        print(f"         HF: {len(hf_ids)} tokens, {hf_time:.3f}ms")
        print(f"         CK: {len(ck_ids)} tokens, {ck_time:.3f}ms ({speedup:.1f}x faster)")

        if not match and len(hf_ids) < 20:
            print(f"         HF ids: {hf_ids}")
            print(f"         CK ids: {ck_ids}")
            # Find first mismatch
            for i, (h, c) in enumerate(zip(hf_ids, ck_ids)):
                if h != c:
                    hf_token = hf_tok.decode([h])
                    ck_token = hf_tok.decode([c]) if c < len(hf_tok) else "?"
                    print(f"         First mismatch at pos {i}: HF={h} ({repr(hf_token)}) vs CK={c} ({repr(ck_token)})")
                    break

    return match


def benchmark(ck_tok, hf_tok, texts: list, n_runs: int = 100):
    """Benchmark C-Kernel vs HuggingFace on multiple texts."""
    print(f"\n  Benchmark ({n_runs} runs each):")
    print("  " + "-" * 60)

    total_ck_time = 0
    total_hf_time = 0
    total_tokens = 0

    for text in texts:
        text_bytes = text.encode('utf-8')
        max_ids = len(text_bytes) * 4 + 100

        # Warm up
        for _ in range(5):
            ids_array = (ctypes.c_int32 * max_ids)()
            lib.ck_tokenizer_encode(ck_tok, text_bytes, len(text_bytes), ids_array, max_ids)
            hf_tok.encode(text, add_special_tokens=False)

        # Benchmark C-Kernel
        ck_start = time.perf_counter()
        for _ in range(n_runs):
            ids_array = (ctypes.c_int32 * max_ids)()
            num_ids = lib.ck_tokenizer_encode(ck_tok, text_bytes, len(text_bytes), ids_array, max_ids)
        ck_time = (time.perf_counter() - ck_start) * 1000

        # Benchmark HuggingFace
        hf_start = time.perf_counter()
        for _ in range(n_runs):
            hf_ids = hf_tok.encode(text, add_special_tokens=False)
        hf_time = (time.perf_counter() - hf_start) * 1000

        speedup = hf_time / ck_time if ck_time > 0 else float('inf')
        avg_ck = ck_time / n_runs
        avg_hf = hf_time / n_runs

        print(f"    Text ({len(text)} chars, {num_ids} tokens):")
        print(f"      C-Kernel: {avg_ck:.4f}ms/encode")
        print(f"      HuggingFace: {avg_hf:.4f}ms/encode")
        print(f"      Speedup: {speedup:.1f}x")

        total_ck_time += ck_time
        total_hf_time += hf_time
        total_tokens += num_ids * n_runs

    print("  " + "-" * 60)
    overall_speedup = total_hf_time / total_ck_time if total_ck_time > 0 else float('inf')
    print(f"  Overall: {overall_speedup:.1f}x faster ({total_tokens} tokens total)")


def main():
    parser = argparse.ArgumentParser(description="Tokenizer parity test")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct",
                       help="HuggingFace model to test against")
    parser.add_argument("--quick", action="store_true",
                       help="Skip large texts for quick test")
    parser.add_argument("--ascii-only", action="store_true",
                       help="Only test ASCII text (full parity expected)")
    parser.add_argument("--all", action="store_true",
                       help="Include UTF-8 tests (partial parity - byte-level BPE needed)")
    args = parser.parse_args()

    print("=" * 70)
    print("C-Kernel Tokenizer: Full Parity Test vs HuggingFace")
    print("=" * 70)
    print(f"\nModel: {args.model}")

    # Load HuggingFace tokenizer
    print("\n1. Loading HuggingFace tokenizer...")
    try:
        from transformers import AutoTokenizer
        hf_tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        print(f"  Vocabulary size: {len(hf_tok.get_vocab())}")
    except Exception as e:
        print(f"  ERROR: Could not load HuggingFace tokenizer: {e}")
        return 1

    # Create C-Kernel tokenizer
    print("\n2. Creating C-Kernel tokenizer...")
    ck_tok = lib.ck_tokenizer_create(CK_TOKENIZER_BPE)
    if not ck_tok:
        print("  ERROR: Failed to create C-Kernel tokenizer")
        return 1

    # Load vocabulary from HuggingFace into C tokenizer
    print("\n3. Loading vocabulary into C-Kernel tokenizer...")
    num_loaded = load_hf_vocab_into_c_tokenizer(ck_tok, hf_tok)

    # Enable trie mode for better performance
    lib.ck_tokenizer_set_use_trie(ck_tok, True)
    print("  Trie mode enabled")

    # Detect space prefix style from vocabulary
    detected_style = lib.ck_tokenizer_detect_space_prefix_style(ck_tok)
    style_names = {CK_SPACE_PREFIX_AUTO: "AUTO", CK_SPACE_PREFIX_GPT2: "GPT-2 (Ġ)", CK_SPACE_PREFIX_SPM: "SentencePiece (▁)"}
    print(f"  Space prefix style: {style_names.get(detected_style, f'Unknown({detected_style})')}")

    # Set special token IDs
    unk_id = hf_tok.unk_token_id if hf_tok.unk_token_id is not None else -1
    bos_id = hf_tok.bos_token_id if hf_tok.bos_token_id is not None else -1
    eos_id = hf_tok.eos_token_id if hf_tok.eos_token_id is not None else -1
    pad_id = hf_tok.pad_token_id if hf_tok.pad_token_id is not None else -1
    lib.ck_tokenizer_set_special_ids(ck_tok, unk_id, bos_id, eos_id, pad_id, -1)
    print(f"  Special tokens: UNK={unk_id}, BOS={bos_id}, EOS={eos_id}, PAD={pad_id}")

    # Test texts organized by category
    print("\n4. Testing encoding parity...")

    # ASCII texts - FULL PARITY expected
    ascii_tests = [
        ("Hello world", "Simple greeting"),
        ("The quick brown fox jumps over the lazy dog.", "Pangram"),
        ("Machine learning is a subset of artificial intelligence.", "Technical text"),
        ("This is a test of the C-Kernel tokenizer's ability to handle various types of text.", "Long sentence"),
        ("1 + 2 = 3", "Math expression"),
        ("Hello, World! How are you?", "Punctuation"),
    ]

    # Code texts - may have slight differences due to tabs/newlines
    code_tests = [
        ("def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)", "Python function"),
        ("for (int i = 0; i < n; i++) { sum += arr[i]; }", "C code"),
    ]

    # UTF-8 texts - PARTIAL PARITY (requires byte-level BPE)
    utf8_tests = [
        ("你好世界", "Chinese"),
        ("Привет мир", "Russian"),
        ("🎉 Happy New Year! 🎊", "Emojis"),
        ("café résumé naïve", "French accents"),
    ]

    if not args.quick:
        ascii_tests.extend([
            ("The transformer architecture has revolutionized natural language processing. "
             "Originally introduced in 2017 by Vaswani et al., the architecture uses "
             "self-attention mechanisms to process sequential data in parallel.", "Long technical"),
        ])
        code_tests.extend([
            ("```python\nimport torch\nfrom transformers import AutoTokenizer\n\n"
             "def tokenize_text(text):\n    tokenizer = AutoTokenizer.from_pretrained('gpt2')\n"
             "    return tokenizer.encode(text)\n```", "Code block"),
        ])

    # Select tests based on flags
    test_texts = []

    print("  [ASCII Tests - Full parity expected]")
    for text, desc in ascii_tests:
        test_texts.append((text, desc, "ascii"))

    if not args.ascii_only:
        print("  [Code Tests - May have minor differences]")
        for text, desc in code_tests:
            test_texts.append((text, desc, "code"))

    if args.all:
        print("  [UTF-8 Tests - Partial parity (byte-level BPE needed)]")
        for text, desc in utf8_tests:
            test_texts.append((text, desc, "utf8"))

    passed = 0
    failed = 0
    ascii_passed = 0
    ascii_total = 0

    for text, desc, category in test_texts:
        result = compare_encoding(ck_tok, hf_tok, text, show_details=True)
        if result:
            passed += 1
            if category == "ascii":
                ascii_passed += 1
        else:
            failed += 1
        if category == "ascii":
            ascii_total += 1

    print(f"\n  Results: {passed}/{passed+failed} tests passed")
    print(f"  ASCII parity: {ascii_passed}/{ascii_total} ({100*ascii_passed//ascii_total if ascii_total else 0}%)")

    # Benchmark
    print("\n5. Performance Benchmark...")
    benchmark_texts = [
        "Hello world",
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence that focuses on the development of algorithms.",
    ]
    benchmark(ck_tok, hf_tok, benchmark_texts, n_runs=100)

    # Cleanup
    lib.ck_tokenizer_free(ck_tok)

    print("\n" + "=" * 70)
    if failed == 0:
        print("ALL PARITY TESTS PASSED")
    else:
        print(f"PARITY TESTS: {failed} FAILED")
    print("=" * 70)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

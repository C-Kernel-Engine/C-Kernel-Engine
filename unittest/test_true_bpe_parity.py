#!/usr/bin/env python3
"""
True BPE Parity Test: C-Kernel True BPE vs HuggingFace

This test verifies that the true BPE implementation achieves 100% parity
with HuggingFace tokenizers by:
1. Loading vocabulary from HuggingFace
2. Loading merge rules from HuggingFace
3. Comparing encoding results

Usage:
    python unittest/test_true_bpe_parity.py [--model MODEL]
"""

import sys
import time
import ctypes
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "unittest"))

# Load the C tokenizer library
try:
    lib = ctypes.CDLL(str(ROOT / "build" / "libckernel_tokenizer.so"))
except Exception as e:
    print(f"ERROR: Could not load tokenizer library: {e}")
    print("Run 'make tokenizer' first")
    sys.exit(1)

# ═══════════════════════════════════════════════════════════════════════════════
# True BPE API bindings
# ═══════════════════════════════════════════════════════════════════════════════

lib.ck_true_bpe_create.restype = ctypes.c_void_p
lib.ck_true_bpe_create.argtypes = []

lib.ck_true_bpe_free.restype = None
lib.ck_true_bpe_free.argtypes = [ctypes.c_void_p]

lib.ck_true_bpe_add_token.restype = ctypes.c_int
lib.ck_true_bpe_add_token.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int32, ctypes.c_float]

lib.ck_true_bpe_add_merge.restype = ctypes.c_int
lib.ck_true_bpe_add_merge.argtypes = [ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32]

lib.ck_true_bpe_add_merge_by_tokens.restype = ctypes.c_int
lib.ck_true_bpe_add_merge_by_tokens.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int32]

lib.ck_true_bpe_set_special_ids.restype = None
lib.ck_true_bpe_set_special_ids.argtypes = [ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32]

lib.ck_true_bpe_encode.restype = ctypes.c_int
lib.ck_true_bpe_encode.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int]

lib.ck_true_bpe_detect_space_style.restype = ctypes.c_int
lib.ck_true_bpe_detect_space_style.argtypes = [ctypes.c_void_p]

lib.ck_true_bpe_vocab_size.restype = ctypes.c_size_t
lib.ck_true_bpe_vocab_size.argtypes = [ctypes.c_void_p]

lib.ck_true_bpe_num_merges.restype = ctypes.c_int32
lib.ck_true_bpe_num_merges.argtypes = [ctypes.c_void_p]


def load_hf_tokenizer(model_name):
    """Load HuggingFace tokenizer."""
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)


def load_vocab_into_true_bpe(bpe, hf_tok):
    """Load vocabulary from HuggingFace tokenizer."""
    vocab = hf_tok.get_vocab()
    print(f"  Loading {len(vocab)} tokens from HuggingFace...")

    start = time.perf_counter()
    loaded = 0
    failed = 0

    for token_str, token_id in vocab.items():
        try:
            token_bytes = token_str.encode('utf-8')
            ret = lib.ck_true_bpe_add_token(bpe, token_bytes, token_id, 0.0)
            if ret == 0:
                loaded += 1
            else:
                failed += 1
        except Exception as e:
            failed += 1
            if failed < 5:
                print(f"    Warning: Could not add token {repr(token_str)}: {e}")

    elapsed = time.perf_counter() - start
    print(f"  Loaded {loaded} tokens in {elapsed:.2f}s ({loaded/elapsed:.0f} tokens/sec)")
    if failed > 0:
        print(f"  Warning: {failed} tokens failed to load")

    return loaded


def load_merges_into_true_bpe(bpe, hf_tok):
    """Load BPE merge rules from HuggingFace tokenizer."""
    # Try to get merges from tokenizer file
    try:
        from transformers.utils import cached_file
        tokenizer_file = cached_file(hf_tok.name_or_path, "tokenizer.json")
        with open(tokenizer_file, 'r') as f:
            tok_json = json.load(f)
    except Exception as e:
        print(f"  Warning: Could not load tokenizer.json: {e}")
        return 0

    # Extract merges from tokenizer.json
    merges = []
    if 'model' in tok_json and 'merges' in tok_json['model']:
        merges = tok_json['model']['merges']
    elif 'merges' in tok_json:
        merges = tok_json['merges']

    if not merges:
        print("  Warning: No merges found in tokenizer.json")
        return 0

    print(f"  Loading {len(merges)} merge rules...")
    start = time.perf_counter()
    loaded = 0
    failed = 0

    for priority, merge in enumerate(merges):
        try:
            # Merge format is "token1 token2"
            if isinstance(merge, str):
                parts = merge.split(' ', 1)
                if len(parts) != 2:
                    failed += 1
                    continue
                left, right = parts
            elif isinstance(merge, (list, tuple)) and len(merge) == 2:
                left, right = merge
            else:
                failed += 1
                continue

            left_bytes = left.encode('utf-8')
            right_bytes = right.encode('utf-8')

            ret = lib.ck_true_bpe_add_merge_by_tokens(bpe, left_bytes, right_bytes, priority)
            if ret == 0:
                loaded += 1
            else:
                failed += 1
        except Exception as e:
            failed += 1
            if failed < 5:
                print(f"    Warning: Could not add merge {repr(merge)}: {e}")

    elapsed = time.perf_counter() - start
    print(f"  Loaded {loaded} merge rules in {elapsed:.2f}s")
    if failed > 0:
        print(f"  Warning: {failed} merges failed to load")

    return loaded


def compare_encoding(bpe, hf_tok, text: str, show_details: bool = True) -> bool:
    """Compare encoding between True BPE and HuggingFace."""
    text_bytes = text.encode('utf-8')
    max_ids = len(text_bytes) * 4 + 100

    # HuggingFace encode
    hf_start = time.perf_counter()
    hf_ids = hf_tok.encode(text, add_special_tokens=False)
    hf_time = (time.perf_counter() - hf_start) * 1000

    # True BPE encode
    ids_array = (ctypes.c_int32 * max_ids)()
    ck_start = time.perf_counter()
    num_ids = lib.ck_true_bpe_encode(bpe, text_bytes, len(text_bytes), ids_array, max_ids)
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

        if not match and len(hf_ids) < 30:
            print(f"         HF ids: {hf_ids}")
            print(f"         CK ids: {ck_ids}")
            # Find first mismatch
            for i in range(max(len(hf_ids), len(ck_ids))):
                h = hf_ids[i] if i < len(hf_ids) else None
                c = ck_ids[i] if i < len(ck_ids) else None
                if h != c:
                    hf_token = hf_tok.decode([h]) if h is not None else "END"
                    ck_token = hf_tok.decode([c]) if c is not None and c < len(hf_tok) else "?"
                    print(f"         First mismatch at pos {i}: HF={h} ({repr(hf_token)}) vs CK={c} ({repr(ck_token)})")
                    break

    return match


def main():
    parser = argparse.ArgumentParser(description="True BPE parity test")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct",
                       help="HuggingFace model to test against")
    args = parser.parse_args()

    print("=" * 70)
    print("True BPE Parity Test: C-Kernel vs HuggingFace")
    print("=" * 70)
    print(f"\nModel: {args.model}")

    # Load HuggingFace tokenizer
    print("\n1. Loading HuggingFace tokenizer...")
    try:
        hf_tok = load_hf_tokenizer(args.model)
        print(f"  Vocabulary size: {len(hf_tok.get_vocab())}")
    except Exception as e:
        print(f"  ERROR: Could not load HuggingFace tokenizer: {e}")
        return 1

    # Create True BPE tokenizer
    print("\n2. Creating True BPE tokenizer...")
    bpe = lib.ck_true_bpe_create()
    if not bpe:
        print("  ERROR: Failed to create True BPE tokenizer")
        return 1

    # Load vocabulary
    print("\n3. Loading vocabulary...")
    num_tokens = load_vocab_into_true_bpe(bpe, hf_tok)
    if num_tokens == 0:
        print("  ERROR: No tokens loaded")
        lib.ck_true_bpe_free(bpe)
        return 1

    # Load merge rules
    print("\n4. Loading merge rules...")
    num_merges = load_merges_into_true_bpe(bpe, hf_tok)

    # Detect space style
    style = lib.ck_true_bpe_detect_space_style(bpe)
    style_names = {0: "AUTO", 1: "GPT-2 (G)", 2: "SentencePiece (_)"}
    print(f"  Space prefix style: {style_names.get(style, f'Unknown({style})')}")

    # Set special token IDs
    unk_id = hf_tok.unk_token_id if hf_tok.unk_token_id is not None else -1
    bos_id = hf_tok.bos_token_id if hf_tok.bos_token_id is not None else -1
    eos_id = hf_tok.eos_token_id if hf_tok.eos_token_id is not None else -1
    pad_id = hf_tok.pad_token_id if hf_tok.pad_token_id is not None else -1
    lib.ck_true_bpe_set_special_ids(bpe, unk_id, bos_id, eos_id, pad_id)
    print(f"  Special tokens: UNK={unk_id}, BOS={bos_id}, EOS={eos_id}, PAD={pad_id}")

    # Print stats
    print(f"\n  True BPE Stats:")
    print(f"    Vocabulary: {lib.ck_true_bpe_vocab_size(bpe)} tokens")
    print(f"    Merges: {lib.ck_true_bpe_num_merges(bpe)} rules")

    # Test texts
    print("\n5. Testing encoding parity...")

    test_texts = [
        ("Hello", "Single word"),
        ("Hello world", "Two words"),
        ("The quick brown fox jumps over the lazy dog.", "Pangram"),
        ("Machine learning is a subset of artificial intelligence.", "Technical"),
        ("1 + 2 = 3", "Math"),
        ("Hello, World! How are you?", "Punctuation"),
        ("def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)", "Python"),
        ("for (int i = 0; i < n; i++) { sum += arr[i]; }", "C code"),
    ]

    passed = 0
    failed = 0

    for text, desc in test_texts:
        result = compare_encoding(bpe, hf_tok, text)
        if result:
            passed += 1
        else:
            failed += 1

    print(f"\n  Results: {passed}/{passed+failed} tests passed ({100*passed//(passed+failed)}%)")

    # Cleanup
    lib.ck_true_bpe_free(bpe)

    print("\n" + "=" * 70)
    if failed == 0:
        print("ALL PARITY TESTS PASSED - 100% PARITY ACHIEVED!")
    else:
        print(f"PARITY TESTS: {failed} FAILED")
        print("Note: Failures may be due to missing merge rules or byte-level BPE handling")
    print("=" * 70)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

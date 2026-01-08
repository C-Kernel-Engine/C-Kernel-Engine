#!/usr/bin/env python3
"""
Multi-Model Tokenizer Parity Test

Compares C-Kernel tokenizer against multiple HuggingFace models:
- GPT-2 (BPE, GPT-2 style)
- Qwen (BPE, GPT-2 style)
- TinyLlama (SentencePiece, LLaMA style)
- Mistral (SentencePiece)
- BERT (WordPiece)

Downloads tokenizers from HuggingFace (just vocab, not model weights).

Usage:
    python unittest/test_tokenizer_multi_model.py
    python unittest/test_tokenizer_multi_model.py --quick      # Fewer tests
    python unittest/test_tokenizer_multi_model.py --verbose    # Show all results
"""

import sys
import os
import time
import ctypes
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "unittest"))

# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════

MODELS = [
    # (name, hf_model_id, tokenizer_type, description)
    ("GPT-2", "openai-community/gpt2", "BPE", "Classic GPT-2 BPE tokenizer"),
    ("Qwen2.5", "Qwen/Qwen2.5-0.5B-Instruct", "BPE", "Qwen BPE tokenizer"),
    ("TinyLlama", "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "SPM", "LLaMA-style SentencePiece"),
    ("Mistral-7B", "mistralai/Mistral-7B-v0.1", "SPM", "Mistral SentencePiece"),
    ("BERT", "google-bert/bert-base-uncased", "WordPiece", "BERT WordPiece tokenizer"),
]

# Test texts organized by category
TEST_TEXTS = {
    "basic": [
        ("Hello world", "Simple greeting"),
        ("The quick brown fox jumps over the lazy dog.", "Pangram"),
        ("Hello, World! How are you?", "With punctuation"),
    ],
    "technical": [
        ("Machine learning is a subset of artificial intelligence.", "ML text"),
        ("The transformer architecture uses self-attention mechanisms.", "AI text"),
        ("Python is a popular programming language.", "Programming"),
    ],
    "code": [
        ("def hello(): return 'world'", "Python function"),
        ("for (int i = 0; i < n; i++) { sum += arr[i]; }", "C code"),
        ("import numpy as np", "Python import"),
    ],
    "numbers": [
        ("1 + 2 = 3", "Simple math"),
        ("The year is 2024.", "Year"),
        ("Price: $99.99", "Currency"),
    ],
    "mixed": [
        ("Hello world! This is a test of the tokenizer.", "Mixed sentence"),
        ("The API returned a 404 error code.", "Tech mixed"),
    ],
}

# ═══════════════════════════════════════════════════════════════════════════════
# Data Classes
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class TestResult:
    text: str
    description: str
    hf_ids: List[int]
    ck_ids: List[int]
    match: bool
    hf_time_ms: float
    ck_time_ms: float
    speedup: float

@dataclass
class ModelResult:
    name: str
    hf_model_id: str
    tokenizer_type: str
    space_style: str
    vocab_size: int
    total_tests: int
    passed_tests: int
    parity_pct: float
    avg_speedup: float
    results: List[TestResult]
    error: Optional[str] = None

# ═══════════════════════════════════════════════════════════════════════════════
# C Library Loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_c_tokenizer():
    """Load the C tokenizer library."""
    lib_path = ROOT / "build" / "libckernel_tokenizer.so"
    if not lib_path.exists():
        raise FileNotFoundError(
            f"Tokenizer library not found: {lib_path}\n"
            "Run 'make tokenizer' first."
        )

    lib = ctypes.CDLL(str(lib_path))

    # Configure function signatures
    lib.ck_tokenizer_create.restype = ctypes.c_void_p
    lib.ck_tokenizer_create.argtypes = [ctypes.c_int]

    lib.ck_tokenizer_free.restype = None
    lib.ck_tokenizer_free.argtypes = [ctypes.c_void_p]

    lib.ck_tokenizer_add_token.restype = ctypes.c_int
    lib.ck_tokenizer_add_token.argtypes = [
        ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int32, ctypes.c_float
    ]

    lib.ck_tokenizer_encode.restype = ctypes.c_int
    lib.ck_tokenizer_encode.argtypes = [
        ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int,
        ctypes.POINTER(ctypes.c_int32), ctypes.c_int
    ]

    lib.ck_tokenizer_set_use_trie.restype = None
    lib.ck_tokenizer_set_use_trie.argtypes = [ctypes.c_void_p, ctypes.c_bool]

    lib.ck_tokenizer_set_special_ids.restype = None
    lib.ck_tokenizer_set_special_ids.argtypes = [
        ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32,
        ctypes.c_int32, ctypes.c_int32, ctypes.c_int32
    ]

    lib.ck_tokenizer_detect_space_prefix_style.restype = ctypes.c_int
    lib.ck_tokenizer_detect_space_prefix_style.argtypes = [ctypes.c_void_p]

    return lib

CK_TOKENIZER_BPE = 0
CK_TOKENIZER_WORDPIECE = 1
CK_TOKENIZER_SPM = 2

SPACE_STYLE_NAMES = {
    0: "AUTO",
    1: "GPT-2 (Ġ)",
    2: "SPM (▁)",
}

# ═══════════════════════════════════════════════════════════════════════════════
# Tokenizer Testing
# ═══════════════════════════════════════════════════════════════════════════════

def load_hf_vocab_into_c(lib, ck_tok, hf_tokenizer) -> int:
    """Load HuggingFace vocabulary into C tokenizer."""
    vocab = hf_tokenizer.get_vocab()
    loaded = 0

    for token_str, token_id in vocab.items():
        try:
            token_bytes = token_str.encode('utf-8')
            ret = lib.ck_tokenizer_add_token(ck_tok, token_bytes, token_id, 0.0)
            if ret == 0:
                loaded += 1
        except:
            pass

    return loaded


def test_single_model(
    lib,
    model_name: str,
    hf_model_id: str,
    tokenizer_type: str,
    test_texts: Dict[str, List[Tuple[str, str]]],
    verbose: bool = False
) -> ModelResult:
    """Test a single model's tokenizer against C-Kernel."""

    # Load HuggingFace tokenizer
    try:
        from transformers import AutoTokenizer
        hf_tok = AutoTokenizer.from_pretrained(hf_model_id, trust_remote_code=True)
    except Exception as e:
        return ModelResult(
            name=model_name,
            hf_model_id=hf_model_id,
            tokenizer_type=tokenizer_type,
            space_style="N/A",
            vocab_size=0,
            total_tests=0,
            passed_tests=0,
            parity_pct=0.0,
            avg_speedup=0.0,
            results=[],
            error=str(e)
        )

    vocab_size = len(hf_tok.get_vocab())

    # Create C tokenizer
    # Always use BPE type - our tokenizer uses greedy longest-match with
    # auto-detected space prefix style (GPT-2 Ġ vs SentencePiece ▁)
    # The tokenizer_type is just for reporting purposes
    ck_tok = lib.ck_tokenizer_create(CK_TOKENIZER_BPE)

    if not ck_tok:
        return ModelResult(
            name=model_name,
            hf_model_id=hf_model_id,
            tokenizer_type=tokenizer_type,
            space_style="N/A",
            vocab_size=vocab_size,
            total_tests=0,
            passed_tests=0,
            parity_pct=0.0,
            avg_speedup=0.0,
            results=[],
            error="Failed to create C tokenizer"
        )

    # Load vocabulary
    load_hf_vocab_into_c(lib, ck_tok, hf_tok)

    # Enable trie mode
    lib.ck_tokenizer_set_use_trie(ck_tok, True)

    # Detect space style
    space_style_id = lib.ck_tokenizer_detect_space_prefix_style(ck_tok)
    space_style = SPACE_STYLE_NAMES.get(space_style_id, f"Unknown({space_style_id})")

    # Set special token IDs
    unk_id = hf_tok.unk_token_id if hf_tok.unk_token_id is not None else -1
    bos_id = hf_tok.bos_token_id if hf_tok.bos_token_id is not None else -1
    eos_id = hf_tok.eos_token_id if hf_tok.eos_token_id is not None else -1
    pad_id = hf_tok.pad_token_id if hf_tok.pad_token_id is not None else -1
    lib.ck_tokenizer_set_special_ids(ck_tok, unk_id, bos_id, eos_id, pad_id, -1)

    # Run tests
    results = []
    total_speedup = 0.0

    for category, texts in test_texts.items():
        for text, desc in texts:
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
            total_speedup += speedup

            results.append(TestResult(
                text=text,
                description=desc,
                hf_ids=hf_ids,
                ck_ids=ck_ids,
                match=match,
                hf_time_ms=hf_time,
                ck_time_ms=ck_time,
                speedup=speedup
            ))

    # Cleanup
    lib.ck_tokenizer_free(ck_tok)

    # Calculate stats
    total_tests = len(results)
    passed_tests = sum(1 for r in results if r.match)
    parity_pct = (passed_tests / total_tests * 100) if total_tests > 0 else 0.0
    avg_speedup = total_speedup / total_tests if total_tests > 0 else 0.0

    return ModelResult(
        name=model_name,
        hf_model_id=hf_model_id,
        tokenizer_type=tokenizer_type,
        space_style=space_style,
        vocab_size=vocab_size,
        total_tests=total_tests,
        passed_tests=passed_tests,
        parity_pct=parity_pct,
        avg_speedup=avg_speedup,
        results=results
    )

# ═══════════════════════════════════════════════════════════════════════════════
# Report Printing
# ═══════════════════════════════════════════════════════════════════════════════

def print_summary_table(results: List[ModelResult]):
    """Print summary comparison table."""
    print()
    print("=" * 100)
    print("  MULTI-MODEL TOKENIZER PARITY SUMMARY")
    print("=" * 100)
    print()

    # Header
    print(f"  {'Model':<15} {'Type':<12} {'Space Style':<12} {'Vocab':<10} {'Parity':<10} {'Speedup':<10} {'Status':<10}")
    print(f"  {'-'*95}")

    for r in results:
        if r.error:
            status = f"\033[91mERROR\033[0m"
            parity = "N/A"
            speedup = "N/A"
            vocab = "N/A"
        else:
            if r.parity_pct >= 80:
                status = f"\033[92mGOOD\033[0m"
            elif r.parity_pct >= 50:
                status = f"\033[93mPARTIAL\033[0m"
            else:
                status = f"\033[91mLOW\033[0m"

            parity = f"{r.parity_pct:.0f}% ({r.passed_tests}/{r.total_tests})"
            speedup = f"{r.avg_speedup:.1f}x"
            vocab = f"{r.vocab_size:,}"

        print(f"  {r.name:<15} {r.tokenizer_type:<12} {r.space_style:<12} {vocab:<10} {parity:<10} {speedup:<10} {status}")

    print()


def print_detailed_results(results: List[ModelResult], verbose: bool = False):
    """Print detailed results for each model."""

    for model_result in results:
        print()
        print(f"  {'─'*80}")
        print(f"  {model_result.name} ({model_result.hf_model_id})")
        print(f"  {'─'*80}")

        if model_result.error:
            print(f"  \033[91mERROR: {model_result.error}\033[0m")
            continue

        print(f"  Type: {model_result.tokenizer_type}  |  Space: {model_result.space_style}  |  Vocab: {model_result.vocab_size:,}")
        print(f"  Parity: {model_result.parity_pct:.1f}%  |  Avg Speedup: {model_result.avg_speedup:.1f}x")
        print()

        if verbose:
            # Show all test results
            for r in model_result.results:
                status = "\033[92mPASS\033[0m" if r.match else "\033[91mFAIL\033[0m"
                text_preview = r.text[:40] + "..." if len(r.text) > 40 else r.text
                print(f"    [{status}] \"{text_preview}\"")
                print(f"           HF: {len(r.hf_ids)} tokens, {r.hf_time_ms:.3f}ms")
                print(f"           CK: {len(r.ck_ids)} tokens, {r.ck_time_ms:.3f}ms ({r.speedup:.1f}x)")
                if not r.match and len(r.hf_ids) < 15:
                    print(f"           HF ids: {r.hf_ids}")
                    print(f"           CK ids: {r.ck_ids}")
        else:
            # Show only failures
            failures = [r for r in model_result.results if not r.match]
            if failures:
                print(f"  Failed tests ({len(failures)}):")
                for r in failures[:5]:  # Show first 5 failures
                    text_preview = r.text[:40] + "..." if len(r.text) > 40 else r.text
                    print(f"    - \"{text_preview}\"")
                    print(f"      HF: {r.hf_ids[:10]}{'...' if len(r.hf_ids) > 10 else ''}")
                    print(f"      CK: {r.ck_ids[:10]}{'...' if len(r.ck_ids) > 10 else ''}")
                if len(failures) > 5:
                    print(f"    ... and {len(failures) - 5} more failures")
            else:
                print(f"  \033[92mAll {model_result.total_tests} tests passed!\033[0m")


def print_analysis(results: List[ModelResult]):
    """Print analysis of results."""
    print()
    print("=" * 100)
    print("  ANALYSIS")
    print("=" * 100)
    print()

    # Group by tokenizer type
    bpe_results = [r for r in results if r.tokenizer_type == "BPE" and not r.error]
    spm_results = [r for r in results if r.tokenizer_type == "SPM" and not r.error]
    wp_results = [r for r in results if r.tokenizer_type == "WordPiece" and not r.error]

    if bpe_results:
        avg_parity = sum(r.parity_pct for r in bpe_results) / len(bpe_results)
        print(f"  BPE Models (GPT-2 style):      {avg_parity:.0f}% avg parity")

    if spm_results:
        avg_parity = sum(r.parity_pct for r in spm_results) / len(spm_results)
        print(f"  SentencePiece Models:          {avg_parity:.0f}% avg parity")

    if wp_results:
        avg_parity = sum(r.parity_pct for r in wp_results) / len(wp_results)
        print(f"  WordPiece Models:              {avg_parity:.0f}% avg parity")

    print()
    print("  Notes:")
    print("  - Parity gaps are primarily due to greedy longest-match vs true BPE merge order")
    print("  - GPT-2 style uses Ġ (U+0120) for space prefix")
    print("  - SentencePiece uses ▁ (U+2581) for space prefix")
    print("  - WordPiece uses ## for subword continuation")
    print()

# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Multi-model tokenizer parity test")
    parser.add_argument("--quick", action="store_true", help="Use fewer test texts")
    parser.add_argument("--verbose", action="store_true", help="Show all test results")
    parser.add_argument("--models", nargs="+", help="Specific models to test (by name)")
    args = parser.parse_args()

    print()
    print("=" * 100)
    print("  C-Kernel Tokenizer: Multi-Model Parity Test")
    print("=" * 100)
    print()

    # Load C library
    print("  Loading C tokenizer library...")
    try:
        lib = load_c_tokenizer()
        print("  Library loaded successfully.")
    except Exception as e:
        print(f"  \033[91mERROR: {e}\033[0m")
        return 1

    # Select test texts
    if args.quick:
        test_texts = {"basic": TEST_TEXTS["basic"]}
        print("  Running quick test (basic texts only)")
    else:
        test_texts = TEST_TEXTS
        print(f"  Running full test ({sum(len(v) for v in TEST_TEXTS.values())} texts)")

    # Select models
    models_to_test = MODELS
    if args.models:
        models_to_test = [m for m in MODELS if m[0] in args.models]
        if not models_to_test:
            print(f"  \033[91mNo matching models found. Available: {[m[0] for m in MODELS]}\033[0m")
            return 1

    print(f"  Testing {len(models_to_test)} models: {[m[0] for m in models_to_test]}")
    print()

    # Test each model
    results = []
    for name, hf_model_id, tok_type, desc in models_to_test:
        print(f"  Testing {name}...", end=" ", flush=True)
        result = test_single_model(lib, name, hf_model_id, tok_type, test_texts, args.verbose)
        if result.error:
            print(f"\033[91mERROR\033[0m")
        else:
            print(f"\033[92m{result.parity_pct:.0f}% parity\033[0m ({result.avg_speedup:.1f}x faster)")
        results.append(result)

    # Print reports
    print_summary_table(results)
    print_detailed_results(results, args.verbose)
    print_analysis(results)

    # Final status
    successful = [r for r in results if not r.error]
    if successful:
        avg_parity = sum(r.parity_pct for r in successful) / len(successful)
        avg_speedup = sum(r.avg_speedup for r in successful) / len(successful)
        print(f"  Overall: {avg_parity:.0f}% average parity, {avg_speedup:.1f}x average speedup")

    print()
    print("=" * 100)

    # Return success if at least one model tested successfully
    return 0 if successful else 1


if __name__ == "__main__":
    sys.exit(main())

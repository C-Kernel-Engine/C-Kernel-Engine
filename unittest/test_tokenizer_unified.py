#!/usr/bin/env python3
"""
Unified Tokenizer Test Suite for C-Kernel-Engine

Comprehensive tests covering:
1. Foundation Tests: Custom vocabulary, UTF-8, emojis, round-trip
2. True BPE Parity: 100% parity with HuggingFace using merge rules
3. Performance: Hash table vs Trie vs Python comparisons
4. Multi-Model: Test against GPT-2, Qwen, LLaMA, BERT, etc.

Usage:
    python unittest/test_tokenizer_unified.py              # Full test suite
    python unittest/test_tokenizer_unified.py --quick      # Quick mode
    python unittest/test_tokenizer_unified.py --section 1  # Foundation only
    python unittest/test_tokenizer_unified.py --section 2  # True BPE only
    python unittest/test_tokenizer_unified.py --verbose    # Detailed output

By Anthony Shivakumar
"""

import sys
import os
import time
import ctypes
import argparse
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple, Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "unittest"))

# ANSI color codes
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"


# ═══════════════════════════════════════════════════════════════════════════════
# Data Classes
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class TestCase:
    """Single test case result."""
    name: str
    description: str
    input_text: str
    passed: bool
    expected: str = ""
    actual: str = ""
    ck_time_ms: float = 0.0
    hf_time_ms: float = 0.0
    speedup: float = 0.0
    ck_tokens: List[int] = field(default_factory=list)
    hf_tokens: Optional[List[int]] = None


@dataclass
class TestSection:
    """Test section with multiple test cases."""
    name: str
    description: str
    tests: List[TestCase] = field(default_factory=list)
    passed: int = 0
    failed: int = 0
    total_time_ms: float = 0.0

    def add_result(self, test: TestCase):
        self.tests.append(test)
        if test.passed:
            self.passed += 1
        else:
            self.failed += 1
        self.total_time_ms += test.ck_time_ms


@dataclass
class PerformanceResult:
    """Performance benchmark result."""
    name: str
    text_length: int
    hash_time_ms: float
    trie_time_ms: float
    hf_time_ms: Optional[float]
    trie_vs_hash: float
    trie_vs_hf: Optional[float]


# ═══════════════════════════════════════════════════════════════════════════════
# C Library Loading
# ═══════════════════════════════════════════════════════════════════════════════

class CTokenizerLib:
    """Wrapper for C tokenizer library."""

    def __init__(self):
        self.lib = None
        self.lib_path = None

    def load(self):
        """Load the tokenizer library."""
        for name in ["libckernel_tokenizer.so", "libckernel_engine.so"]:
            for path in [ROOT / name, ROOT / "build" / name]:
                if path.exists():
                    self.lib_path = path
                    break
            if self.lib_path:
                break

        if not self.lib_path:
            raise FileNotFoundError(
                "Could not find tokenizer library.\n"
                "Run 'make tokenizer' first."
            )

        self.lib = ctypes.CDLL(str(self.lib_path))
        self._setup_signatures()
        return self

    def _setup_signatures(self):
        """Set up C function signatures."""
        lib = self.lib

        # Base tokenizer API
        lib.ck_tokenizer_create.restype = ctypes.c_void_p
        lib.ck_tokenizer_create.argtypes = [ctypes.c_int]

        lib.ck_tokenizer_free.restype = None
        lib.ck_tokenizer_free.argtypes = [ctypes.c_void_p]

        lib.ck_tokenizer_add_token.restype = ctypes.c_int
        lib.ck_tokenizer_add_token.argtypes = [
            ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int32, ctypes.c_float
        ]

        lib.ck_tokenizer_add_special_token.restype = ctypes.c_int
        lib.ck_tokenizer_add_special_token.argtypes = [
            ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int32
        ]

        lib.ck_tokenizer_encode.restype = ctypes.c_int
        lib.ck_tokenizer_encode.argtypes = [
            ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int,
            ctypes.POINTER(ctypes.c_int32), ctypes.c_int
        ]

        lib.ck_tokenizer_decode.restype = ctypes.c_int
        lib.ck_tokenizer_decode.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32), ctypes.c_int,
            ctypes.c_char_p, ctypes.c_int
        ]

        lib.ck_tokenizer_set_use_trie.restype = None
        lib.ck_tokenizer_set_use_trie.argtypes = [ctypes.c_void_p, ctypes.c_bool]

        lib.ck_tokenizer_lookup.restype = ctypes.c_int32
        lib.ck_tokenizer_lookup.argtypes = [ctypes.c_void_p, ctypes.c_char_p]

        # True BPE API
        lib.ck_true_bpe_create.restype = ctypes.c_void_p
        lib.ck_true_bpe_create.argtypes = []

        lib.ck_true_bpe_free.restype = None
        lib.ck_true_bpe_free.argtypes = [ctypes.c_void_p]

        lib.ck_true_bpe_add_token.restype = ctypes.c_int
        lib.ck_true_bpe_add_token.argtypes = [
            ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int32, ctypes.c_float
        ]

        lib.ck_true_bpe_add_merge_by_tokens.restype = ctypes.c_int
        lib.ck_true_bpe_add_merge_by_tokens.argtypes = [
            ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int32
        ]

        lib.ck_true_bpe_set_special_ids.restype = None
        lib.ck_true_bpe_set_special_ids.argtypes = [
            ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32,
            ctypes.c_int32, ctypes.c_int32
        ]

        lib.ck_true_bpe_encode.restype = ctypes.c_int
        lib.ck_true_bpe_encode.argtypes = [
            ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int,
            ctypes.c_void_p, ctypes.c_int
        ]

        lib.ck_true_bpe_detect_space_style.restype = ctypes.c_int
        lib.ck_true_bpe_detect_space_style.argtypes = [ctypes.c_void_p]

        lib.ck_true_bpe_vocab_size.restype = ctypes.c_size_t
        lib.ck_true_bpe_vocab_size.argtypes = [ctypes.c_void_p]

        lib.ck_true_bpe_num_merges.restype = ctypes.c_int32
        lib.ck_true_bpe_num_merges.argtypes = [ctypes.c_void_p]


# ═══════════════════════════════════════════════════════════════════════════════
# Section 1: Foundation Tests (Custom Vocabulary)
# ═══════════════════════════════════════════════════════════════════════════════

def run_foundation_tests(clib: CTokenizerLib, verbose: bool = False) -> TestSection:
    """Run foundation tests with custom vocabulary."""
    section = TestSection(
        name="Foundation Tests",
        description="Custom vocabulary: ASCII, UTF-8, emojis, round-trip"
    )

    lib = clib.lib

    # Create tokenizer
    tok = lib.ck_tokenizer_create(0)  # BPE type

    # Add special tokens
    lib.ck_tokenizer_add_special_token(tok, b"<unk>", 0)
    lib.ck_tokenizer_add_special_token(tok, b"<s>", 1)
    lib.ck_tokenizer_add_special_token(tok, b"</s>", 2)

    # Add basic vocabulary
    vocab = {
        "hello": 100, "world": 101, "\u0120world": 102,  # Ġworld
        "the": 110, "quick": 111, "brown": 112, "fox": 113,
        "\u0120the": 114, "\u0120quick": 115, "\u0120brown": 116, "\u0120fox": 117,
        # UTF-8
        "cafe": 200, "naive": 201,
        "\u65e5\u672c\u8a9e": 203,  # Japanese
        "\u4e16\u754c": 204,        # Chinese
        # Emojis
        "\U0001f389": 205,  # Party
        "\U0001f680": 206,  # Rocket
    }

    for token, id in vocab.items():
        lib.ck_tokenizer_add_token(tok, token.encode('utf-8'), id, 0.0)

    lib.ck_tokenizer_set_use_trie(tok, True)

    def encode(text):
        text_bytes = text.encode('utf-8')
        max_ids = len(text_bytes) * 4 + 10
        ids = (ctypes.c_int32 * max_ids)()
        num = lib.ck_tokenizer_encode(tok, text_bytes, len(text_bytes), ids, max_ids)
        return list(ids[:num])

    # Test cases: (input, expected_ids, description)
    test_cases = [
        # ASCII
        ("hello", [100], "Single ASCII word"),
        ("hello world", [100, 102], "Two words with space"),
        ("helloworld", [100, 101], "Concatenated words"),

        # Space handling
        ("the quick brown fox", [110, 115, 116, 117], "Pangram subset"),

        # UTF-8 - Note: these may not match exactly since vocab uses simplified
        ("\u65e5\u672c\u8a9e", [203], "Japanese text"),
        ("\u4e16\u754c", [204], "Chinese text"),

        # Emojis
        ("\U0001f389", [205], "Party emoji"),
        ("\U0001f680", [206], "Rocket emoji"),

        # Edge cases
        ("", [], "Empty string"),
    ]

    for input_text, expected, desc in test_cases:
        start = time.perf_counter()
        actual = encode(input_text)
        elapsed = (time.perf_counter() - start) * 1000

        passed = actual == expected
        section.add_result(TestCase(
            name=desc,
            description=desc,
            input_text=input_text,
            passed=passed,
            expected=str(expected),
            actual=str(actual),
            ck_time_ms=elapsed,
            ck_tokens=actual
        ))

    lib.ck_tokenizer_free(tok)
    return section


# ═══════════════════════════════════════════════════════════════════════════════
# Section 2: True BPE Parity Tests
# ═══════════════════════════════════════════════════════════════════════════════

def run_true_bpe_parity_tests(clib: CTokenizerLib, model: str = "Qwen/Qwen2.5-0.5B-Instruct", verbose: bool = False) -> TestSection:
    """Run True BPE parity tests against HuggingFace."""
    section = TestSection(
        name="True BPE Parity",
        description=f"100% parity with HuggingFace ({model})"
    )

    lib = clib.lib

    # Try to load HuggingFace tokenizer
    try:
        from transformers import AutoTokenizer
        from transformers.utils import cached_file
        hf_tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    except ImportError:
        section.add_result(TestCase(
            name="HuggingFace Import",
            description="transformers package required",
            input_text="",
            passed=False,
            expected="transformers installed",
            actual="Import failed - run: pip install transformers"
        ))
        return section
    except Exception as e:
        section.add_result(TestCase(
            name="HuggingFace Load",
            description=f"Load {model}",
            input_text="",
            passed=False,
            expected="Model loaded",
            actual=str(e)
        ))
        return section

    # Create True BPE tokenizer
    bpe = lib.ck_true_bpe_create()
    if not bpe:
        section.add_result(TestCase(
            name="Create True BPE",
            description="Create tokenizer",
            input_text="",
            passed=False,
            expected="Tokenizer created",
            actual="Failed to create"
        ))
        return section

    # Load vocabulary
    vocab = hf_tok.get_vocab()
    loaded = 0
    for token_str, token_id in vocab.items():
        try:
            ret = lib.ck_true_bpe_add_token(bpe, token_str.encode('utf-8'), token_id, 0.0)
            if ret == 0:
                loaded += 1
        except:
            pass

    # Load merge rules
    merges_loaded = 0
    try:
        tokenizer_file = cached_file(hf_tok.name_or_path, "tokenizer.json")
        with open(tokenizer_file, 'r') as f:
            tok_json = json.load(f)

        merges = []
        if 'model' in tok_json and 'merges' in tok_json['model']:
            merges = tok_json['model']['merges']
        elif 'merges' in tok_json:
            merges = tok_json['merges']

        for priority, merge in enumerate(merges):
            try:
                if isinstance(merge, str):
                    parts = merge.split(' ', 1)
                    if len(parts) == 2:
                        left, right = parts
                        ret = lib.ck_true_bpe_add_merge_by_tokens(
                            bpe, left.encode('utf-8'), right.encode('utf-8'), priority
                        )
                        if ret == 0:
                            merges_loaded += 1
            except:
                pass
    except:
        pass

    # Set special tokens
    unk_id = hf_tok.unk_token_id if hf_tok.unk_token_id is not None else -1
    bos_id = hf_tok.bos_token_id if hf_tok.bos_token_id is not None else -1
    eos_id = hf_tok.eos_token_id if hf_tok.eos_token_id is not None else -1
    pad_id = hf_tok.pad_token_id if hf_tok.pad_token_id is not None else -1
    lib.ck_true_bpe_set_special_ids(bpe, unk_id, bos_id, eos_id, pad_id)

    # Test texts
    test_texts = [
        ("Hello", "Single word"),
        ("Hello world", "Two words"),
        ("The quick brown fox jumps over the lazy dog.", "Pangram"),
        ("Machine learning is a subset of artificial intelligence.", "Technical"),
        ("1 + 2 = 3", "Math expression"),
        ("Hello, World! How are you?", "Punctuation"),
        ("def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)", "Python code"),
        ("for (int i = 0; i < n; i++) { sum += arr[i]; }", "C code"),
    ]

    for text, desc in test_texts:
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

        section.add_result(TestCase(
            name=desc,
            description=f"{desc} ({len(text)} chars)",
            input_text=text[:50] + "..." if len(text) > 50 else text,
            passed=match,
            expected=f"{len(hf_ids)} tokens: {hf_ids[:5]}..." if len(hf_ids) > 5 else str(hf_ids),
            actual=f"{len(ck_ids)} tokens: {ck_ids[:5]}..." if len(ck_ids) > 5 else str(ck_ids),
            ck_time_ms=ck_time,
            hf_time_ms=hf_time,
            speedup=speedup,
            ck_tokens=ck_ids,
            hf_tokens=hf_ids
        ))

    lib.ck_true_bpe_free(bpe)
    return section


# ═══════════════════════════════════════════════════════════════════════════════
# Section 3: Performance Benchmarks
# ═══════════════════════════════════════════════════════════════════════════════

def run_performance_tests(clib: CTokenizerLib, verbose: bool = False) -> Tuple[TestSection, List[PerformanceResult]]:
    """Run performance benchmarks: Hash vs Trie vs Python."""
    section = TestSection(
        name="Performance Benchmarks",
        description="Hash Table vs Trie vs Python comparison"
    )
    perf_results = []

    lib = clib.lib

    # Create tokenizer with large vocab
    tok = lib.ck_tokenizer_create(0)
    for i in range(2000):
        lib.ck_tokenizer_add_token(tok, f"word{i}".encode(), 1000 + i, 0.0)
        lib.ck_tokenizer_add_token(tok, f"the{i}".encode(), 3000 + i, 0.0)

    def encode_with_mode(text, use_trie):
        lib.ck_tokenizer_set_use_trie(tok, use_trie)
        text_bytes = text.encode('utf-8')
        max_ids = len(text_bytes) * 4 + 10
        ids = (ctypes.c_int32 * max_ids)()
        lib.ck_tokenizer_encode(tok, text_bytes, len(text_bytes), ids, max_ids)

    # Try to get HuggingFace for comparison
    hf_tok = None
    try:
        from transformers import AutoTokenizer
        hf_tok = AutoTokenizer.from_pretrained("gpt2")
    except:
        pass

    test_cases = [
        ("Short", "hello world " * 5),
        ("Medium", "word0 word1 word2 word3 word4 " * 20),
        ("Long", "word0 word1 word2 word3 word4 " * 100),
        ("Very Long", "word0 word1 word2 word3 word4 " * 500),
    ]

    for name, text in test_cases:
        n_runs = 100 if len(text) < 1000 else 50

        # Warmup
        for _ in range(5):
            encode_with_mode(text, False)
            encode_with_mode(text, True)

        # Hash table
        start = time.perf_counter()
        for _ in range(n_runs):
            encode_with_mode(text, False)
        hash_time = (time.perf_counter() - start) * 1000 / n_runs

        # Trie
        start = time.perf_counter()
        for _ in range(n_runs):
            encode_with_mode(text, True)
        trie_time = (time.perf_counter() - start) * 1000 / n_runs

        # HuggingFace (if available)
        hf_time = None
        if hf_tok:
            start = time.perf_counter()
            for _ in range(n_runs):
                hf_tok.encode(text)
            hf_time = (time.perf_counter() - start) * 1000 / n_runs

        trie_vs_hash = hash_time / trie_time if trie_time > 0 else 0
        trie_vs_hf = hf_time / trie_time if (hf_time and trie_time > 0) else None

        perf_results.append(PerformanceResult(
            name=name,
            text_length=len(text),
            hash_time_ms=hash_time,
            trie_time_ms=trie_time,
            hf_time_ms=hf_time,
            trie_vs_hash=trie_vs_hash,
            trie_vs_hf=trie_vs_hf
        ))

        section.add_result(TestCase(
            name=f"Perf: {name}",
            description=f"{name} text ({len(text)} chars)",
            input_text=text[:30] + "...",
            passed=True,  # Performance tests always pass
            expected="Benchmark completed",
            actual=f"Hash={hash_time:.3f}ms, Trie={trie_time:.3f}ms ({trie_vs_hash:.1f}x)",
            ck_time_ms=trie_time
        ))

    lib.ck_tokenizer_free(tok)
    return section, perf_results


# ═══════════════════════════════════════════════════════════════════════════════
# Report Printing
# ═══════════════════════════════════════════════════════════════════════════════

def print_header():
    """Print test header."""
    print()
    print(f"{BOLD}{'='*100}{RESET}")
    print(f"{BOLD}  C-KERNEL-ENGINE UNIFIED TOKENIZER TEST SUITE{RESET}")
    print(f"{BOLD}{'='*100}{RESET}")
    print()


def print_section(section: TestSection, verbose: bool = False):
    """Print test section results."""
    status_color = GREEN if section.failed == 0 else RED
    status_text = "PASS" if section.failed == 0 else "FAIL"

    print()
    print(f"{BOLD}{CYAN}  [{section.name}]{RESET}")
    print(f"  {section.description}")
    print(f"  {'─'*96}")
    print()

    # Table header
    print(f"    {'#':<3} {'Test':<35} {'Input':<25} {'Status':<8} {'Time (ms)':<12} {'Speedup':<10}")
    print(f"    {'-'*95}")

    for i, test in enumerate(section.tests, 1):
        status = f"{GREEN}PASS{RESET}" if test.passed else f"{RED}FAIL{RESET}"
        input_display = test.input_text[:23] + ".." if len(test.input_text) > 25 else test.input_text
        input_display = input_display.replace('\n', '\\n') if input_display else "(empty)"

        speedup_str = f"{test.speedup:.1f}x" if test.speedup > 0 else "-"

        print(f"    {i:<3} {test.name[:35]:<35} {input_display:<25} {status:<8} {test.ck_time_ms:<12.3f} {speedup_str:<10}")

        # Show details for failures
        if not test.passed and verbose:
            print(f"         {RED}Expected: {test.expected}{RESET}")
            print(f"         {RED}Actual:   {test.actual}{RESET}")

    # Section summary
    print()
    print(f"    {'-'*60}")
    pct = section.passed / (section.passed + section.failed) * 100 if (section.passed + section.failed) > 0 else 0
    print(f"    Results: {section.passed}/{section.passed + section.failed} passed ({pct:.0f}%)  |  Time: {section.total_time_ms:.2f}ms")
    print()


def print_performance_table(perf_results: List[PerformanceResult]):
    """Print performance comparison table."""
    if not perf_results:
        return

    print()
    print(f"{BOLD}{CYAN}  [Performance Comparison]{RESET}")
    print(f"  Hash Table vs Trie vs Python (HuggingFace)")
    print(f"  {'─'*96}")
    print()

    print(f"    {'Test':<12} {'Chars':<8} {'Hash (ms)':<12} {'Trie (ms)':<12} {'HF (ms)':<12} {'Trie/Hash':<12} {'Trie/HF':<12}")
    print(f"    {'-'*80}")

    for p in perf_results:
        hf_str = f"{p.hf_time_ms:.3f}" if p.hf_time_ms else "N/A"
        t_h_str = f"{GREEN}{p.trie_vs_hash:.2f}x{RESET}" if p.trie_vs_hash >= 1.0 else f"{RED}{p.trie_vs_hash:.2f}x{RESET}"
        t_hf_str = f"{GREEN}{p.trie_vs_hf:.1f}x{RESET}" if p.trie_vs_hf and p.trie_vs_hf >= 1.0 else ("N/A" if not p.trie_vs_hf else f"{p.trie_vs_hf:.1f}x")

        print(f"    {p.name:<12} {p.text_length:<8} {p.hash_time_ms:<12.3f} {p.trie_time_ms:<12.3f} {hf_str:<12} {t_h_str:<12} {t_hf_str:<12}")

    print()


def print_summary(sections: List[TestSection]):
    """Print overall summary."""
    total_passed = sum(s.passed for s in sections)
    total_failed = sum(s.failed for s in sections)
    total_tests = total_passed + total_failed
    total_time = sum(s.total_time_ms for s in sections)

    print()
    print(f"{BOLD}{'='*100}{RESET}")
    print(f"{BOLD}  SUMMARY{RESET}")
    print(f"{BOLD}{'='*100}{RESET}")
    print()

    for section in sections:
        pct = section.passed / (section.passed + section.failed) * 100 if (section.passed + section.failed) > 0 else 0
        status = f"{GREEN}PASS{RESET}" if section.failed == 0 else f"{RED}FAIL{RESET}"
        print(f"    {section.name:<30} {section.passed:>3}/{section.passed + section.failed:<3} ({pct:>5.1f}%)  {status}")

    print(f"    {'─'*60}")
    overall_pct = total_passed / total_tests * 100 if total_tests > 0 else 0
    overall_status = f"{GREEN}ALL TESTS PASSED{RESET}" if total_failed == 0 else f"{RED}{total_failed} TESTS FAILED{RESET}"
    print(f"    {'TOTAL':<30} {total_passed:>3}/{total_tests:<3} ({overall_pct:>5.1f}%)  {overall_status}")
    print()
    print(f"    Total Time: {total_time:.2f}ms")
    print()
    print(f"{BOLD}{'='*100}{RESET}")
    print()


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Unified Tokenizer Test Suite")
    parser.add_argument("--quick", action="store_true", help="Quick mode (fewer tests)")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--section", type=int, choices=[1, 2, 3],
                       help="Run specific section: 1=Foundation, 2=True BPE, 3=Performance")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct",
                       help="HuggingFace model for True BPE tests")
    args = parser.parse_args()

    print_header()

    # Load C library
    print(f"  Loading C tokenizer library...")
    try:
        clib = CTokenizerLib().load()
        print(f"  {GREEN}Library loaded:{RESET} {clib.lib_path}")
    except Exception as e:
        print(f"  {RED}ERROR: {e}{RESET}")
        return 1

    sections = []
    perf_results = []

    # Section 1: Foundation Tests
    if args.section is None or args.section == 1:
        print(f"\n  Running Foundation Tests...")
        section = run_foundation_tests(clib, args.verbose)
        sections.append(section)
        print_section(section, args.verbose)

    # Section 2: True BPE Parity Tests
    if args.section is None or args.section == 2:
        print(f"\n  Running True BPE Parity Tests ({args.model})...")
        section = run_true_bpe_parity_tests(clib, args.model, args.verbose)
        sections.append(section)
        print_section(section, args.verbose)

    # Section 3: Performance Benchmarks
    if (args.section is None or args.section == 3) and not args.quick:
        print(f"\n  Running Performance Benchmarks...")
        section, perf_results = run_performance_tests(clib, args.verbose)
        sections.append(section)
        print_section(section, args.verbose)
        print_performance_table(perf_results)

    # Summary
    print_summary(sections)

    # Return code
    total_failed = sum(s.failed for s in sections)
    return 0 if total_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

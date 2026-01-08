#!/usr/bin/env python3
"""
llama.cpp Tokenizer Comparison Test

Compares C-Kernel True BPE tokenizer against llama.cpp's tokenizer
to verify parity with the reference implementation.

Usage:
    python unittest/test_tokenizer_llamacpp.py              # Full test
    python unittest/test_tokenizer_llamacpp.py --verbose    # Detailed output
    python unittest/test_tokenizer_llamacpp.py --model PATH # Custom GGUF model

By Anthony Shivakumar
"""

import sys
import os
import time
import ctypes
import argparse
import subprocess
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]

# ANSI color codes
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"


@dataclass
class ComparisonResult:
    """Single comparison result."""
    text: str
    ck_tokens: List[int]
    llama_tokens: List[int]
    passed: bool
    ck_time_ms: float = 0.0
    llama_time_ms: float = 0.0
    speedup: float = 0.0


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

        lib.ck_true_bpe_vocab_size.restype = ctypes.c_size_t
        lib.ck_true_bpe_vocab_size.argtypes = [ctypes.c_void_p]

        lib.ck_true_bpe_num_merges.restype = ctypes.c_int32
        lib.ck_true_bpe_num_merges.argtypes = [ctypes.c_void_p]


def find_llama_tokenize() -> Optional[Path]:
    """Find llama-tokenize binary."""
    paths = [
        ROOT / "llama.cpp" / "build" / "bin" / "llama-tokenize",
        ROOT / "llama.cpp" / "llama-tokenize",
        Path("/usr/local/bin/llama-tokenize"),
    ]
    for p in paths:
        if p.exists():
            return p
    return None


def find_gguf_model() -> Optional[Path]:
    """Find a GGUF model file."""
    # Prefer Qwen for consistency with our HuggingFace tests
    gguf_files = [
        ROOT / "qwen2.5-3b-instruct-q4_k_m.gguf",
        ROOT / "SmolLM-1.7B-Instruct.Q4_K_M.gguf",
    ]
    for p in gguf_files:
        if p.exists():
            return p
    # Search for any .gguf
    for p in ROOT.glob("*.gguf"):
        if p.stat().st_size < 5_000_000_000:  # < 5GB
            return p
    return None


def tokenize_with_llama(llama_bin: Path, model_path: Path, text: str) -> Tuple[List[int], float]:
    """
    Tokenize text using llama.cpp's tokenizer.
    Returns (token_ids, time_ms).
    """
    start = time.perf_counter()
    result = subprocess.run(
        [
            str(llama_bin),
            "-m", str(model_path),
            "-p", text,
            "--ids",
            "--no-bos",
            "--log-disable"
        ],
        capture_output=True,
        text=True
    )
    elapsed = (time.perf_counter() - start) * 1000

    if result.returncode != 0:
        raise RuntimeError(f"llama-tokenize failed: {result.stderr}")

    # Parse output like "[9707, 1879]"
    output = result.stdout.strip()
    if output.startswith("[") and output.endswith("]"):
        tokens = json.loads(output)
        return tokens, elapsed
    return [], elapsed


def create_ck_bpe_tokenizer(clib: CTokenizerLib, hf_model: str = "Qwen/Qwen2.5-3B-Instruct"):
    """Create and initialize C-Kernel True BPE tokenizer from HuggingFace."""
    lib = clib.lib
    bpe = lib.ck_true_bpe_create()

    if not bpe:
        raise RuntimeError("Failed to create True BPE tokenizer")

    # Load from HuggingFace
    try:
        from transformers import AutoTokenizer
        hf_tok = AutoTokenizer.from_pretrained(hf_model, trust_remote_code=True)
    except ImportError:
        raise ImportError("transformers package required: pip install transformers")

    # Load vocabulary
    vocab = hf_tok.get_vocab()
    for token_str, token_id in vocab.items():
        try:
            token_bytes = token_str.encode('utf-8')
            lib.ck_true_bpe_add_token(bpe, token_bytes, token_id, 0.0)
        except:
            pass

    # Load merges
    try:
        from transformers.utils import cached_file
        tokenizer_json_path = cached_file(hf_model, "tokenizer.json")
        with open(tokenizer_json_path, "r", encoding="utf-8") as f:
            tokenizer_json = json.load(f)

        if "model" in tokenizer_json and "merges" in tokenizer_json["model"]:
            merges = tokenizer_json["model"]["merges"]
            for priority, merge in enumerate(merges):
                parts = merge.split(" ", 1)
                if len(parts) == 2:
                    left, right = parts
                    lib.ck_true_bpe_add_merge_by_tokens(
                        bpe,
                        left.encode('utf-8'),
                        right.encode('utf-8'),
                        priority
                    )
    except Exception as e:
        print(f"{YELLOW}Warning: Could not load merges: {e}{RESET}")

    # Set special token IDs
    unk_id = vocab.get("<|endoftext|>", -1)
    bos_id = vocab.get("<|im_start|>", -1)
    eos_id = vocab.get("<|im_end|>", -1)
    pad_id = vocab.get("<|endoftext|>", -1)
    lib.ck_true_bpe_set_special_ids(bpe, unk_id, bos_id, eos_id, pad_id)

    return bpe


def tokenize_with_ck(clib: CTokenizerLib, bpe, text: str) -> Tuple[List[int], float]:
    """
    Tokenize text using C-Kernel True BPE.
    Returns (token_ids, time_ms).
    """
    lib = clib.lib
    text_bytes = text.encode('utf-8')
    max_tokens = len(text_bytes) * 4 + 10
    ids = (ctypes.c_int32 * max_tokens)()

    start = time.perf_counter()
    num_tokens = lib.ck_true_bpe_encode(bpe, text_bytes, len(text_bytes), ids, max_tokens)
    elapsed = (time.perf_counter() - start) * 1000

    return list(ids[:num_tokens]), elapsed


def print_header():
    """Print test header."""
    print()
    print(f"{BOLD}{'='*70}{RESET}")
    print(f"{BOLD}   C-Kernel vs llama.cpp Tokenizer Comparison{RESET}")
    print(f"{BOLD}{'='*70}{RESET}")
    print()


def print_result(result: ComparisonResult, verbose: bool = False):
    """Print a single comparison result."""
    status = f"{GREEN}PASS{RESET}" if result.passed else f"{RED}FAIL{RESET}"

    # Truncate long text
    display_text = result.text[:40] + "..." if len(result.text) > 40 else result.text
    display_text = display_text.replace("\n", "\\n")

    if result.passed:
        speedup_str = f"{result.speedup:6.1f}x" if result.speedup > 0 else "  N/A "
        print(f"  [{status}] {display_text:45} {speedup_str}")
    else:
        print(f"  [{status}] {display_text:45}")

    if verbose or not result.passed:
        if not result.passed:
            print(f"        CK:    {result.ck_tokens}")
            print(f"        llama: {result.llama_tokens}")


def run_comparison_tests(
    clib: CTokenizerLib,
    llama_bin: Path,
    model_path: Path,
    verbose: bool = False
) -> List[ComparisonResult]:
    """Run comparison tests between C-Kernel and llama.cpp."""

    print(f"  Model: {model_path.name}")
    print(f"  llama-tokenize: {llama_bin}")
    print()

    # Create C-Kernel tokenizer
    print(f"  Loading C-Kernel True BPE tokenizer...")
    bpe = create_ck_bpe_tokenizer(clib, "Qwen/Qwen2.5-3B-Instruct")
    vocab_size = clib.lib.ck_true_bpe_vocab_size(bpe)
    num_merges = clib.lib.ck_true_bpe_num_merges(bpe)
    print(f"  Loaded: {vocab_size:,} tokens, {num_merges:,} merges")
    print()

    # Test cases
    test_cases = [
        # Basic
        "Hello",
        "Hello world",
        "The quick brown fox",

        # Punctuation
        "Hello, world!",
        "What's up?",
        "I'm happy.",

        # Code
        "def fibonacci(n):",
        "int main() { return 0; }",
        "print('hello')",

        # Numbers
        "12345",
        "3.14159",
        "The year is 2024.",

        # Unicode
        "Café",
        "naïve",
        "日本語",
        "中文测试",

        # Mixed
        "Hello, 世界!",
        "Python 3.11 is great!",
        "The price is $99.99",
    ]

    results = []

    print(f"  {BOLD}Running {len(test_cases)} comparison tests...{RESET}")
    print()

    for text in test_cases:
        try:
            ck_tokens, ck_time = tokenize_with_ck(clib, bpe, text)
            llama_tokens, llama_time = tokenize_with_llama(llama_bin, model_path, text)

            passed = ck_tokens == llama_tokens
            speedup = llama_time / ck_time if ck_time > 0 else 0

            result = ComparisonResult(
                text=text,
                ck_tokens=ck_tokens,
                llama_tokens=llama_tokens,
                passed=passed,
                ck_time_ms=ck_time,
                llama_time_ms=llama_time,
                speedup=speedup
            )
            results.append(result)
            print_result(result, verbose)

        except Exception as e:
            result = ComparisonResult(
                text=text,
                ck_tokens=[],
                llama_tokens=[],
                passed=False
            )
            results.append(result)
            print(f"  [{RED}FAIL{RESET}] {text[:40]}: {e}")

    # Cleanup
    clib.lib.ck_true_bpe_free(bpe)

    return results


def print_summary(results: List[ComparisonResult]):
    """Print test summary."""
    passed = sum(1 for r in results if r.passed)
    total = len(results)
    pct = (passed / total * 100) if total > 0 else 0

    print()
    print(f"  {BOLD}{'─'*60}{RESET}")

    # Calculate average speedup for passing tests
    speedups = [r.speedup for r in results if r.passed and r.speedup > 0]
    avg_speedup = sum(speedups) / len(speedups) if speedups else 0

    status_color = GREEN if passed == total else (YELLOW if passed > total // 2 else RED)

    print(f"  {BOLD}Results:{RESET} {status_color}{passed}/{total}{RESET} ({pct:.1f}%) tests passed")

    if avg_speedup > 0:
        print(f"  {BOLD}Average Speedup:{RESET} {CYAN}{avg_speedup:.1f}x{RESET} faster than llama.cpp")

    if passed < total:
        print()
        print(f"  {YELLOW}Failed tests:{RESET}")
        for r in results:
            if not r.passed:
                text_short = r.text[:30] + "..." if len(r.text) > 30 else r.text
                print(f"    - \"{text_short}\"")
                print(f"      CK:    {r.ck_tokens}")
                print(f"      llama: {r.llama_tokens}")

    print()


def main():
    parser = argparse.ArgumentParser(description="Compare C-Kernel vs llama.cpp tokenizer")
    parser.add_argument("--model", "-m", type=str, help="Path to GGUF model")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()

    print_header()

    # Find llama-tokenize
    llama_bin = find_llama_tokenize()
    if not llama_bin:
        print(f"  {RED}ERROR:{RESET} Could not find llama-tokenize binary")
        print(f"        Build llama.cpp first: cd llama.cpp && cmake -B build && cmake --build build")
        return 1

    # Find model
    if args.model:
        model_path = Path(args.model)
        if not model_path.exists():
            print(f"  {RED}ERROR:{RESET} Model not found: {args.model}")
            return 1
    else:
        model_path = find_gguf_model()
        if not model_path:
            print(f"  {RED}ERROR:{RESET} No GGUF model found")
            print(f"        Download a model: huggingface-cli download Qwen/Qwen2.5-3B-Instruct-GGUF qwen2.5-3b-instruct-q4_k_m.gguf")
            return 1

    # Load C library
    try:
        clib = CTokenizerLib().load()
        print(f"  Loaded: {clib.lib_path.name}")
    except FileNotFoundError as e:
        print(f"  {RED}ERROR:{RESET} {e}")
        return 1

    # Run tests
    results = run_comparison_tests(clib, llama_bin, model_path, args.verbose)

    # Summary
    print_summary(results)

    # Exit code
    passed = sum(1 for r in results if r.passed)
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())

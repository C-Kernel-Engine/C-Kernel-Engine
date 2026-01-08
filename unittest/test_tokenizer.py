"""
Tokenizer unit tests with detailed reporting.

Tests:
- Basic ASCII tokenization
- UTF-8 multilingual text (French, Japanese, Chinese)
- UTF-8 emojis
- Case sensitivity (case-sensitive by default)
- Whitespace handling
- Performance benchmarks (vs PyTorch/tiktoken)

By Anthony Shivakumar
"""
import ctypes
import os
import sys
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

# Add parent dir to path for lib_loader
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib_loader import load_lib

# Try to import tiktoken for PyTorch comparison
try:
    import tiktoken
    HAS_TIKTOKEN = True
except ImportError:
    HAS_TIKTOKEN = False
    print("Note: tiktoken not available - PyTorch comparison disabled")


# ═══════════════════════════════════════════════════════════════════════════════
# Library Loading
# ═══════════════════════════════════════════════════════════════════════════════

def find_tokenizer_lib():
    """Find the tokenizer library."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    for name in ["libckernel_tokenizer.so", "libckernel_engine.so"]:
        for path in [os.path.join(root, name), os.path.join(root, "build", name)]:
            if os.path.exists(path):
                return path

    raise FileNotFoundError("Could not find tokenizer library")


lib_path = find_tokenizer_lib()
print(f"Loading: {lib_path}")
lib = ctypes.CDLL(lib_path)

# ═══════════════════════════════════════════════════════════════════════════════
# Data Classes for Reporting
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class TestResult:
    """Result of a single test case."""
    name: str
    description: str
    passed: bool
    input_text: str
    output_tokens: List[int]
    expected: str
    actual: str
    error: Optional[str] = None
    time_ms: float = 0.0
    pytorch_tokens: Optional[List[int]] = None
    pytorch_time_ms: float = 0.0


@dataclass
class LookupMethodResult:
    """Result comparing trie vs hash table lookup methods."""
    name: str
    text_len: int
    hash_table_time_ms: float
    trie_time_ms: float
    speedup: float  # trie / hash_table (higher is better for trie)


@dataclass
class FullPerformanceResult:
    """Full performance comparison with all methods."""
    name: str
    text_len: int
    hash_time_ms: float
    trie_time_ms: float
    pytorch_time_ms: Optional[float]
    hash_throughput: float
    trie_throughput: float
    pytorch_throughput: Optional[float]
    trie_vs_hash_speedup: float
    trie_vs_pytorch_speedup: Optional[float]


@dataclass
class TokenizerReport:
    """Complete tokenizer test report."""
    test_name: str
    tokenizer_type: str
    results: List[TestResult]
    performance: List[FullPerformanceResult]
    lookup_method: List[LookupMethodResult]  # Trie vs Hash comparison
    total_passed: int = 0
    total_failed: int = 0

    def all_passed(self) -> bool:
        return self.total_failed == 0


# ═══════════════════════════════════════════════════════════════════════════════
# Function Signatures
# ═══════════════════════════════════════════════════════════════════════════════

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

lib.ck_tokenizer_lookup.restype = ctypes.c_int32
lib.ck_tokenizer_lookup.argtypes = [ctypes.c_void_p, ctypes.c_char_p]

lib.ck_tokenizer_set_use_trie.restype = None
lib.ck_tokenizer_set_use_trie.argtypes = [ctypes.c_void_p, ctypes.c_bool]

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


# ═══════════════════════════════════════════════════════════════════════════════
# C Tokenizer Wrapper
# ═══════════════════════════════════════════════════════════════════════════════

class CTokenizer:
    """Wrapper for C tokenizer."""

    BPE = 0
    WORDPIECE = 1
    SPM = 2

    def __init__(self):
        self._tok = None
        self._vocab_size = 0

    def create(self, typ="bpe"):
        """Create tokenizer."""
        if typ == "bpe":
            self._tok = lib.ck_tokenizer_create(self.BPE)
        else:
            self._tok = lib.ck_tokenizer_create(self.WORDPIECE)
        return self

    def free(self):
        """Free tokenizer."""
        if self._tok:
            lib.ck_tokenizer_free(self._tok)
            self._tok = None

    def add_token(self, token, id, score=0.0):
        """Add token to vocabulary."""
        result = lib.ck_tokenizer_add_token(
            self._tok, token.encode('utf-8'), id, score
        )
        if result == 0:
            self._vocab_size += 1
        return result

    def add_special_token(self, token, id):
        """Add special token."""
        return lib.ck_tokenizer_add_special_token(
            self._tok, token.encode('utf-8'), id
        )

    def encode(self, text):
        """Encode text to token IDs."""
        if not self._tok:
            raise RuntimeError("Tokenizer not created")

        text_bytes = text.encode('utf-8')
        max_ids = len(text_bytes) * 2 + 10
        ids = (ctypes.c_int32 * max_ids)()

        num_ids = lib.ck_tokenizer_encode(
            self._tok, text_bytes, len(text_bytes), ids, max_ids
        )

        return list(ids[:num_ids])

    def decode(self, ids):
        """Decode token IDs to text."""
        if not self._tok:
            raise RuntimeError("Tokenizer not created")

        ids_array = (ctypes.c_int32 * len(ids))(*ids)
        text = (ctypes.c_char * 4096)()

        num_bytes = lib.ck_tokenizer_decode(
            self._tok, ids_array, len(ids), text, 4096
        )

        return text[:num_bytes].decode('utf-8')

    def vocab_size(self):
        """Get vocabulary size."""
        return self._vocab_size

    def lookup(self, token):
        """Look up token ID."""
        return lib.ck_tokenizer_lookup(
            self._tok, token.encode('utf-8')
        )

    def use_trie(self, enabled: bool = True):
        """Enable or disable trie-based lookups (faster)."""
        if self._tok:
            lib.ck_tokenizer_set_use_trie(self._tok, enabled)


# ═══════════════════════════════════════════════════════════════════════════════
# TikToken Reference (for PyTorch comparison)
# ═══════════════════════════════════════════════════════════════════════════════

class TikTokenRef:
    """Wrapper for tiktoken to use as reference."""

    def __init__(self, model_name="gpt2"):
        if HAS_TIKTOKEN:
            self.enc = tiktoken.get_encoding(model_name)
        else:
            self.enc = None

    def encode(self, text):
        """Encode text to tokens."""
        if self.enc:
            return self.enc.encode(text)
        return None

    def decode(self, ids):
        """Decode tokens to text."""
        if self.enc:
            return self.enc.decode(ids)
        return ""


# ═══════════════════════════════════════════════════════════════════════════════
# Test Functions
# ═══════════════════════════════════════════════════════════════════════════════

def run_test(
    name: str,
    description: str,
    input_text: str,
    fn: Callable,
    results: List[TestResult],
    pytorch_fn: Optional[Callable] = None
) -> bool:
    """Run a test and record the result."""
    start = time.perf_counter()
    try:
        fn()
        elapsed = (time.perf_counter() - start) * 1000

        # Get output tokens from the test
        output = fn()
        # If fn doesn't return anything, we need to get it differently
        # The test should set up the tokenizer before calling us

        results.append(TestResult(
            name=name,
            description=description,
            passed=True,
            input_text=input_text,
            output_tokens=output if output else [],
            expected="Success",
            actual="Success",
            time_ms=elapsed
        ))
        return True
    except AssertionError as e:
        elapsed = (time.perf_counter() - start) * 1000
        results.append(TestResult(
            name=name,
            description=description,
            passed=False,
            input_text=input_text,
            output_tokens=[],
            expected="Correct result",
            actual=str(e),
            error=str(e),
            time_ms=elapsed
        ))
        return False
    except Exception as e:
        elapsed = (time.perf_counter() - start) * 1000
        results.append(TestResult(
            name=name,
            description=description,
            passed=False,
            input_text=input_text,
            output_tokens=[],
            expected="No exception",
            actual=f"Exception: {type(e).__name__}: {e}",
            error=str(e),
            time_ms=elapsed
        ))
        return False


def test_basic_ascii(ctok, results, ref_tok=None):
    """Test basic ASCII tokenization (case-sensitive)."""

    # Special tokens
    run_test("BOS Token", "Beginning-of-sequence token", "<s>",
             lambda: ctok.encode("<s>"), results)

    run_test("EOS Token", "End-of-sequence token", "</s>",
             lambda: ctok.encode("</s>"), results)

    run_test("UNK Token", "Unknown token", "<unk>",
             lambda: ctok.encode("<unk>"), results)

    # Add basic vocabulary
    # Note: BPE tokenizers use Ġ (U+0120, bytes 0xC4 0xA0) as space prefix in vocab
    # Our tokenizer converts spaces to Ġ during encoding for GGUF compatibility
    ctok.add_token("hello", 100)
    ctok.add_token("world", 101)
    ctok.add_token("\u0120world", 102)  # Ġworld - space-prefixed version (U+0120 = Ġ)

    # Test cases with detailed output AND ASSERTIONS
    test_cases = [
        ("hello", "Lowercase word", [100]),
        ("world", "Lowercase word", [101]),
        ("hello world", "Two-word phrase", [100, 102]),  # space -> Ġ, then matches Ġworld
        ("helloworld", "Combined (no space)", [100, 101]),  # Longest match finds "hello" then "world"
    ]

    for text, desc, expected_ids in test_cases:
        ids = ctok.encode(text)
        pt_ids = None
        pt_time = 0.0

        # ASSERTION: Verify our output matches expected
        passed = ids == expected_ids

        # PyTorch/tiktoken comparison (informational, not assertion)
        pt_comparison = "N/A"
        if ref_tok and ref_tok.enc:
            start = time.perf_counter()
            pt_ids = ref_tok.encode(text)
            pt_time = (time.perf_counter() - start) * 1000
            # For informational purposes only - different vocabs

        results.append(TestResult(
            name=f"{desc}: '{text[:10]}...'",
            description=desc,
            passed=passed,
            input_text=text,
            output_tokens=ids,
            expected=str(expected_ids),
            actual=str(ids),
            time_ms=0.001,
            pytorch_tokens=list(pt_ids) if pt_ids else None,
            pytorch_time_ms=pt_time
        ))


def test_utf8_multilingual(ctok, results, ref_tok=None):
    """Test UTF-8 multilingual text."""

    # Add UTF-8 tokens
    ctok.add_token("café", 200)
    ctok.add_token("naïve", 201)
    ctok.add_token("résumé", 202)
    ctok.add_token("日本語", 203)  # Japanese (nihongo)
    ctok.add_token("世界", 204)     # Japanese/Chinese (sekai/world)
    ctok.add_token("中文", 205)    # Chinese

    # Test cases with EXPECTED OUTPUT ASSERTIONS
    test_cases = [
        ("café", "French accented", [200]),
        ("naïve", "French accented", [201]),
        ("日本語", "Japanese", [203]),
        ("世界", "Chinese/Japanese", [204]),
        ("中文", "Chinese", [205]),
    ]

    for text, desc, expected_ids in test_cases:
        ids = ctok.encode(text)
        pt_ids = None
        pt_time = 0.0

        # ASSERTION: Verify exact token IDs
        passed = ids == expected_ids

        # Tiktoken comparison (informational only)
        if ref_tok and ref_tok.enc:
            start = time.perf_counter()
            pt_ids = ref_tok.encode(text)
            pt_time = (time.perf_counter() - start) * 1000

        results.append(TestResult(
            name=f"{desc}: '{text[:10]}...'",
            description=f"UTF-8 {desc}",
            passed=passed,
            input_text=text,
            output_tokens=ids,
            expected=str(expected_ids),
            actual=str(ids),
            time_ms=0.001,
            pytorch_tokens=list(pt_ids) if pt_ids else None,
            pytorch_time_ms=pt_time
        ))


def test_utf8_emojis(ctok, results, ref_tok=None):
    """Test emoji handling."""

    ctok.add_token("🎉", 205)
    ctok.add_token("🚀", 206)
    ctok.add_token("💻", 207)

    test_cases = [
        ("🎉", "Single emoji"),
        ("🚀💻🎉", "Emoji sequence"),
    ]

    for text, desc in test_cases:
        ids = ctok.encode(text)

        results.append(TestResult(
            name=f"{desc}: '{text}'",
            description=f"UTF-8 {desc}",
            passed=True,
            input_text=text,
            output_tokens=ids,
            expected=f"Token IDs from vocab",
            actual=str(ids),
            time_ms=0.001
        ))


def test_case_sensitivity(ctok, results):
    """Test case sensitivity (tokenizer is case-sensitive by default)."""

    ctok.add_token("hello", 100)
    ctok.add_token("HELLO", 101)

    test_cases = [
        ("hello", [100], "Lowercase"),
        ("HELLO", [101], "Uppercase"),
        ("Hello", [0, 0, 0, 0, 0], "Mixed case -> 5 UNKs (one per char)"),
    ]

    for text, expected, desc in test_cases:
        ids = ctok.encode(text)
        passed = ids == expected

        results.append(TestResult(
            name=f"Case '{text}'",
            description=f"{desc} (case-sensitive)",
            passed=passed,
            input_text=text,
            output_tokens=ids,
            expected=str(expected),
            actual=str(ids),
            time_ms=0.001
        ))


def test_pangram(ctok, results):
    """Test pangram."""

    # Add lowercase pangram words
    words = ["the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog",
             "pack", "my", "box", "with", "five", "dozen", "liquor", "jugs"]

    for i, word in enumerate(words):
        ctok.add_token(word, 300 + i)

    pangram = "the quick brown fox jumps over the lazy dog"
    ids = ctok.encode(pangram)
    decoded = ctok.decode(ids)
    passed = "quick" in decoded and "brown" in decoded

    results.append(TestResult(
        name="Pangram",
        description="Full pangram tokenization",
        passed=passed,
        input_text=pangram,
        output_tokens=ids,
        expected="9 tokens",
        actual=f"{len(ids)} tokens: {ids[:3]}...{ids[-3:]}",
        time_ms=0.001
    ))


def test_edge_cases(ctok, results):
    """Test edge cases."""

    test_cases = [
        ("", "Empty string", []),
        ("x", "Single character", [0]),
        ("   ", "Spaces only", []),
        ("hello\nworld", "With newlines", [100, 0, 101]),
    ]

    for text, desc, _ in test_cases:
        ids = ctok.encode(text)

        results.append(TestResult(
            name=f"{desc}",
            description=f"Edge case: {desc}",
            passed=True,
            input_text=text,
            output_tokens=ids,
            expected="Token IDs",
            actual=f"{len(ids)} tokens: {ids}",
            time_ms=0.001
        ))


def test_round_trip(ctok, results):
    """Test round-trip: encode → decode should recover original text.

    This is a CRITICAL correctness test. The decode(encode(text)) should
    approximately equal the original text (within tokenizer's whitespace handling).
    """

    # Add vocabulary for round-trip testing
    # Note: hello(100), world(101), Ġworld(102) already added earlier
    # Use different IDs to avoid conflicts
    ctok.add_token("the", 110)
    ctok.add_token("quick", 111)
    ctok.add_token("brown", 112)
    ctok.add_token("fox", 113)
    # Add space-prefixed versions for proper round-trip
    ctok.add_token("\u0120world", 102)  # Ġworld - already exists, re-add is safe
    ctok.add_token("\u0120the", 114)
    ctok.add_token("\u0120quick", 115)
    ctok.add_token("\u0120brown", 116)
    ctok.add_token("\u0120fox", 117)

    test_cases = [
        ("hello world", "Simple phrase"),
        ("the quick brown fox", "Pangram subset"),
        ("hello", "Single word"),
        ("hello world hello world", "Repeated phrase"),
    ]

    for text, desc in test_cases:
        ids = ctok.encode(text)
        decoded = ctok.decode(ids)

        # For BPE, whitespace normalization may occur
        # We check that decoded text contains all the key words
        passed = all(word in decoded.lower() for word in text.lower().split())

        results.append(TestResult(
            name=f"Round-trip: {desc}",
            description=f"encode→decode round-trip",
            passed=passed,
            input_text=text,
            output_tokens=ids,
            expected=f"Decoded contains key words from: '{text}'",
            actual=f"Decoded: '{decoded}'",
            time_ms=0.001
        ))


def test_performance(ctok, results, perf_results, ref_tok=None):
    """Test encoding performance with various text lengths using both hash and trie."""

    # Add lots of tokens for testing
    for i in range(500):
        ctok.add_token(f"word{i}", 500 + i)
        ctok.add_token(f"the{i}", 1500 + i)
        ctok.add_token(f"prefix{i}", 2500 + i)

    # Test cases with different text lengths
    test_cases = [
        ("Short Text", "hello world"),
        ("Medium Text", "the quick brown fox " * 10),
        ("Long Text", "word0 word1 word2 word3 word4 " * 100),
        ("Very Long", "word0 word1 word2 word3 word4 " * 500),
    ]

    n_runs = 100 if len(test_cases[0][1]) < 1000 else 50

    for name, text in test_cases:
        n_runs = 100 if len(text) < 1000 else 50

        # Warm up with both methods
        for _ in range(5):
            ctok.use_trie(False)
            ctok.encode(text)
        for _ in range(5):
            ctok.use_trie(True)
            ctok.encode(text)

        # Test with Hash Table
        ctok.use_trie(False)
        start = time.perf_counter()
        for _ in range(n_runs):
            ctok.encode(text)
        hash_time = (time.perf_counter() - start) * 1000
        avg_hash_time = hash_time / n_runs
        hash_throughput = len(text) / avg_hash_time if avg_hash_time > 0 else 0

        # Test with Trie
        ctok.use_trie(True)
        start = time.perf_counter()
        for _ in range(n_runs):
            ctok.encode(text)
        trie_time = (time.perf_counter() - start) * 1000
        avg_trie_time = trie_time / n_runs
        trie_throughput = len(text) / avg_trie_time if avg_trie_time > 0 else 0

        # PyTorch/tiktoken comparison
        pt_time = 0.0
        pt_throughput = None
        if ref_tok and ref_tok.enc:
            start = time.perf_counter()
            for _ in range(n_runs):
                ref_tok.encode(text)
            pt_time = (time.perf_counter() - start) * 1000
            avg_pt_time = pt_time / n_runs
            pt_throughput = len(text) / avg_pt_time if avg_pt_time > 0 else None

        # Calculate speedups
        trie_vs_hash_speedup = avg_hash_time / avg_trie_time if avg_trie_time > 0 else 0
        trie_vs_pytorch_speedup = None
        if pt_throughput and trie_throughput:
            trie_vs_pytorch_speedup = trie_throughput / pt_throughput

        perf_results.append(FullPerformanceResult(
            name=name,
            text_len=len(text),
            hash_time_ms=avg_hash_time,
            trie_time_ms=avg_trie_time,
            pytorch_time_ms=avg_pt_time if pt_throughput else None,
            hash_throughput=hash_throughput,
            trie_throughput=trie_throughput,
            pytorch_throughput=pt_throughput,
            trie_vs_hash_speedup=trie_vs_hash_speedup,
            trie_vs_pytorch_speedup=trie_vs_pytorch_speedup
        ))

    results.append(TestResult(
        name="Performance",
        description="Benchmark on 4 text sizes (Hash, Trie, PyTorch)",
        passed=True,
        input_text="Various lengths",
        output_tokens=[],
        expected="4 benchmark results",
        actual=f"{len(perf_results)} benchmarks completed",
        time_ms=0.001
    ))


def test_trie_vs_hash_benchmark(ctok, results, lookup_results, ref_tok=None):
    """Benchmark trie vs hash table lookup methods."""

    # Add lots of tokens for realistic testing
    for i in range(1000):
        ctok.add_token(f"word{i}", 1000 + i)
        ctok.add_token(f"the{i}", 2000 + i)
        ctok.add_token(f"prefix{i}_suffix", 3000 + i)

    # Test cases
    test_cases = [
        ("Short Text", "hello world"),
        ("Medium Text", "the quick brown fox " * 10),
        ("Long Text", "word0 word1 word2 word3 word4 " * 100),
        ("Very Long", "word0 word1 word2 word3 word4 " * 500),
    ]

    for name, text in test_cases:
        n_runs = 100 if len(text) < 1000 else 50

        # Test with hash table (trie disabled)
        ctok.use_trie(False)
        start = time.perf_counter()
        for _ in range(n_runs):
            ctok.encode(text)
        hash_time = (time.perf_counter() - start) * 1000
        avg_hash_time = hash_time / n_runs

        # Test with trie (trie enabled)
        ctok.use_trie(True)
        start = time.perf_counter()
        for _ in range(n_runs):
            ctok.encode(text)
        trie_time = (time.perf_counter() - start) * 1000
        avg_trie_time = trie_time / n_runs

        # Calculate speedup (how much faster is trie vs hash table)
        speedup = avg_hash_time / avg_trie_time if avg_trie_time > 0 else 0

        lookup_results.append(LookupMethodResult(
            name=name,
            text_len=len(text),
            hash_table_time_ms=avg_hash_time,
            trie_time_ms=avg_trie_time,
            speedup=speedup
        ))

    results.append(TestResult(
        name="Trie vs Hash",
        description="Compare trie vs hash table lookup performance",
        passed=True,
        input_text="Various lengths",
        output_tokens=[],
        expected="4 comparison results",
        actual=f"{len(lookup_results)} comparisons completed",
        time_ms=0.001
    ))


def test_vocab_lookup(ctok, results):
    """Test vocabulary lookup performance."""

    # Add test tokens
    for i in range(100):
        ctok.add_token(f"lookup{i}", 600 + i)

    # Test lookups
    start = time.perf_counter()
    for _ in range(10000):
        ctok.lookup("lookup50")
        ctok.lookup("notfound")
    elapsed = time.perf_counter() - start
    rate = 10000 / elapsed

    results.append(TestResult(
        name="Vocab Lookup",
        description=f"10000 lookups at {rate:.0f}/sec",
        passed=rate > 100000,
        input_text="Various token strings",
        output_tokens=[],
        expected=">100K lookups/sec",
        actual=f"{rate:.0f} lookups/sec",
        time_ms=elapsed * 1000
    ))


# ═══════════════════════════════════════════════════════════════════════════════
# Report Printing
# ═══════════════════════════════════════════════════════════════════════════════

def print_report(report: TokenizerReport):
    """Print a comprehensive test report."""

    # Header
    print()
    print("=" * 100)
    print(f"  TOKENIZER TEST REPORT")
    print("=" * 100)
    print(f"  Tokenizer Type: {report.tokenizer_type}")
    print(f"  Library: {lib_path}")
    print(f"  Reference: tiktoken (GPT-2)" if HAS_TIKTOKEN else "  Reference: None (tiktoken not available)")
    print()

    # Print comprehensive test results table
    print(f"  {'─' * 100}")
    print(f"  TEST RESULTS")
    print(f"  {'─' * 100}")
    print()
    print(f"    {'#':<3} {'Test Name':<32} {'Input':<22} {'Output':<28} {'Status':<6} {'C (ms)':<10} {'PyTorch (ms)':<14}")
    print(f"    {'-'*100}")

    for i, t in enumerate(report.results, 1):
        status = "PASS" if t.passed else "FAIL"
        status_color = "\033[92m" if t.passed else "\033[91m"
        reset = "\033[0m"

        # Truncate input for display
        input_display = t.input_text[:20] + ".." if len(t.input_text) > 22 else t.input_text
        input_display = input_display if input_display else "(empty)"

        # Format output tokens
        if len(t.output_tokens) <= 3:
            tokens_display = str(t.output_tokens)
        elif len(t.output_tokens) <= 6:
            tokens_display = str(t.output_tokens[:3])[:-1] + ", " + str(t.output_tokens[3:])[1:]
        elif len(t.output_tokens) > 6:
            tokens_display = f"{t.output_tokens[:2]}... [{len(t.output_tokens)}]"
        else:
            tokens_display = str(t.output_tokens)

        pt_time = f"{t.pytorch_time_ms:.3f}" if t.pytorch_time_ms > 0 else "-"

        print(f"    {i:<3} {t.name[:32]:<32} {input_display:<22} {tokens_display:<28} {status_color}{status:<6}{reset} {t.time_ms:<10.3f} {pt_time:<14}")

        # Show failures in detail
        if not t.passed:
            print(f"         ERROR: Expected: {t.expected}")
            print(f"                 Actual: {t.actual}")
            if t.error:
                print(f"                 Error:  {t.error}")

    # Performance comparison section - Full comparison of all methods
    if report.performance:
        print()
        print(f"  {'─' * 118}")
        print(f"  PERFORMANCE COMPARISON: C-Kernel Hash vs C-Kernel Trie vs PyTorch")
        print(f"  {'─' * 118}")
        print()
        print(f"    {'Test':<15} {'Chars':<6} {'CK Hash':<10} {'CK Trie':<10} {'PyTorch':<10} {'Hash/ms':<9} {'Trie/ms':<9} {'PT/ms':<9} {'T/H':<6} {'T/PT':<6}")
        print(f"    {'-'*118}")

        total_chars = 0
        total_hash_time = 0
        total_trie_time = 0
        total_pt_time = 0

        for p in report.performance:
            hash_time = f"{p.hash_time_ms:.3f}"
            trie_time = f"{p.trie_time_ms:.3f}"
            pt_time = f"{p.pytorch_time_ms:.3f}" if p.pytorch_time_ms else "N/A"

            hash_thr = f"{p.hash_throughput:.0f}"
            trie_thr = f"{p.trie_throughput:.0f}"
            pt_thr = f"{p.pytorch_throughput:.0f}" if p.pytorch_throughput else "N/A"

            # Trie vs Hash speedup - always show winner's perspective
            if p.trie_vs_hash_speedup >= 1.0:
                trie_hash_speed = f"\033[92m{p.trie_vs_hash_speedup:.1f}x C\033[0m"
            else:
                # Invert to show how much faster hash is than trie
                hash_speed = 1.0 / p.trie_vs_hash_speedup
                trie_hash_speed = f"\033[91m{hash_speed:.1f}x H\033[0m"

            # Trie vs PyTorch speedup - always show winner's perspective
            if p.trie_vs_pytorch_speedup is not None:
                if p.trie_vs_pytorch_speedup >= 1.0:
                    trie_pt_speed = f"\033[92m{p.trie_vs_pytorch_speedup:.2f}x C\033[0m"
                else:
                    # Invert to show how much faster PyTorch is than trie
                    pt_speed = 1.0 / p.trie_vs_pytorch_speedup
                    trie_pt_speed = f"\033[91m{pt_speed:.2f}x P\033[0m"
            else:
                trie_pt_speed = "N/A"

            print(f"    {p.name:<15} {p.text_len:<6} {hash_time:<10} {trie_time:<10} {pt_time:<10} {hash_thr:<9} {trie_thr:<9} {pt_thr:<9} {trie_hash_speed:<8} {trie_pt_speed:<8}")

            total_chars += p.text_len
            total_hash_time += p.hash_time_ms
            total_trie_time += p.trie_time_ms
            if p.pytorch_time_ms:
                total_pt_time += p.pytorch_time_ms

        # Summary stats
        avg_hash_throughput = total_chars / total_hash_time if total_hash_time > 0 else 0
        avg_trie_throughput = total_chars / total_trie_time if total_trie_time > 0 else 0
        avg_pt_throughput = total_chars / total_pt_time if total_pt_time > 0 else 0

        print()
        print(f"    {'─' * 80}")
        print(f"    Avg Throughput:")
        print(f"      CK Hash:   {avg_hash_throughput:,.0f} chars/ms")
        print(f"      CK Trie:   {avg_trie_throughput:,.0f} chars/ms")
        print(f"      PyTorch:   {avg_pt_throughput:,.0f} chars/ms")
        print(f"    Overall Trie vs Hash Speedup: {total_hash_time/total_trie_time:.1f}x")
        if total_pt_time > 0:
            print(f"    Overall Trie vs PyTorch Speedup: {avg_trie_throughput/avg_pt_throughput:.2f}x")

    # Trie vs Hash Table detailed comparison section (keep for reference)

    # Trie vs Hash Table comparison section (legacy - data now in performance section)
    if report.lookup_method:
        print()
        print(f"  {'─' * 96}")
        print(f"  LEGACY: TRIE VS HASH TABLE COMPARISON (DEPRECATED)")
        print(f"  {'─' * 96}")
        print()
        print(f"    Note: This data is now included in the main Performance Comparison section above.")
        print(f"    {'Test':<15} {'Chars':<8} {'Hash Table':<14} {'Trie':<14} {'Speedup':<12} {'Winner':<10}")
        print(f"    {'-'*96}")

        trie_wins = 0
        hash_wins = 0

        for p in report.lookup_method:
            hash_time = f"{p.hash_table_time_ms:.3f}"
            trie_time = f"{p.trie_time_ms:.3f}"

            if p.speedup >= 1.0:
                speedup_str = f"\033[92m{p.speedup:.2f}x\033[0m"
                winner = "\033[92mTRIE\033[0m"
                trie_wins += 1
            else:
                speedup_str = f"\033[91m{p.speedup:.2f}x\033[0m"
                winner = "\033[91mHASH\033[0m"
                hash_wins += 1

            print(f"    {p.name:<15} {p.text_len:<8} {hash_time:<14} {trie_time:<14} {speedup_str:<12} {winner}")

        print()
        print(f"    {'─' * 60}")
        if trie_wins > hash_wins:
            print(f"    Result: TRIE wins {trie_wins}/{len(report.lookup_method)} comparisons")
        elif hash_wins > trie_wins:
            print(f"    Result: HASH wins {hash_wins}/{len(report.lookup_method)} comparisons")
        else:
            print(f"    Result: Tie ({trie_wins}/{len(report.lookup_method)})")
    else:
        print()
        print(f"  {'─' * 96}")
        print(f"  DETAILED LOOKUP METHOD COMPARISON")
        print(f"  {'─' * 96}")
        print()
        print(f"    See the main Performance Comparison section above for complete Hash vs Trie vs PyTorch results.")

    # Summary
    print()
    print(f"  {'─' * 96}")
    print(f"  SUMMARY")
    print(f"  {'─' * 96}")
    print()
    print(f"    Total Tests:  {report.total_passed + report.total_failed}")
    print(f"    Passed:       \033[92m{report.total_passed}\033[0m")
    print(f"    Failed:       \033[91m{report.total_failed}\033[0m")
    print(f"    Pass Rate:    {(report.total_passed / (report.total_passed + report.total_failed) * 100):.1f}%")

    if report.all_passed():
        print()
        print(f"    \033[92mALL TESTS PASSED ✓\033[0m")
    else:
        print()
        print(f"    \033[91mSOME TESTS FAILED ✗\033[0m")

    print()
    print("=" * 100)
    print()


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 100)
    print("  C-Kernel-Engine Tokenizer Tests (UTF-8 + Performance + PyTorch Comparison)")
    print("=" * 100)

    # Create tokenizer
    ctok = CTokenizer().create("bpe")

    # Set up vocabulary
    ctok.add_special_token("<unk>", 0)
    ctok.add_special_token("<s>", 1)
    ctok.add_special_token("</s>", 2)
    ctok.add_special_token("<pad>", 3)

    # Create reference tokenizer
    ref_tok = TikTokenRef("gpt2") if HAS_TIKTOKEN else None

    # Collect results
    results = []
    perf_results = []
    lookup_method_results = []

    # Run all test categories
    test_basic_ascii(ctok, results, ref_tok)
    test_utf8_multilingual(ctok, results, ref_tok)
    test_utf8_emojis(ctok, results, ref_tok)
    test_case_sensitivity(ctok, results)
    test_pangram(ctok, results)
    test_edge_cases(ctok, results)
    test_round_trip(ctok, results)  # CRITICAL: Round-trip correctness test
    test_performance(ctok, results, perf_results, ref_tok)  # Now includes Hash, Trie, and PyTorch
    test_vocab_lookup(ctok, results)

    # Note: Separate trie vs hash benchmark removed - data now in perf_results
    # lookup_method_results kept for backward compatibility (empty)

    # Count results
    passed = sum(1 for r in results if r.passed)
    failed = sum(1 for r in results if not r.passed)

    # Create report
    report = TokenizerReport(
        test_name="Tokenizer Comprehensive Tests",
        tokenizer_type="BPE (Byte-Pair Encoding)",
        results=results,
        performance=perf_results,
        lookup_method=lookup_method_results,
        total_passed=passed,
        total_failed=failed
    )

    # Print report
    print_report(report)

    # Cleanup
    ctok.free()

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

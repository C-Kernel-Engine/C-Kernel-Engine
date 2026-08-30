#!/usr/bin/env python3
"""
Test True BPE Tokenizer: Special Token Handling and GPT-2 Byte Decoding

This test verifies:
1. Special tokens like <|im_end|> are encoded as single tokens (not broken into chars)
2. GPT-2 byte-level encoded characters are decoded correctly (Ċ → newline)
3. Chat templates work correctly with special tokens

Tests against HuggingFace for parity when available.
"""

import sys
import ctypes
import json
import os
import unicodedata
from pathlib import Path

# Try to load HuggingFace transformers
try:
    from transformers import AutoTokenizer
    HAS_HF = True
except ImportError:
    HAS_HF = False
    print("Note: transformers not available, skipping HuggingFace comparison")

# Load the C tokenizer library
LIB_PATH = Path(__file__).parent.parent / "build" / "libckernel_tokenizer.so"
if not LIB_PATH.exists():
    print(f"ERROR: Tokenizer library not found: {LIB_PATH}")
    print("Run 'make tokenizer' to build it.")
    sys.exit(1)

lib = ctypes.CDLL(str(LIB_PATH))

# ═══════════════════════════════════════════════════════════════════════════════
# C FUNCTION SIGNATURES
# ═══════════════════════════════════════════════════════════════════════════════

# true_bpe.h functions
lib.ck_true_bpe_create.restype = ctypes.c_void_p
lib.ck_true_bpe_create.argtypes = []

lib.ck_true_bpe_free.restype = None
lib.ck_true_bpe_free.argtypes = [ctypes.c_void_p]

lib.ck_true_bpe_add_token.restype = ctypes.c_int
lib.ck_true_bpe_add_token.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int32, ctypes.c_float]

lib.ck_true_bpe_add_merge.restype = ctypes.c_int
lib.ck_true_bpe_add_merge.argtypes = [ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32]

lib.ck_true_bpe_add_special_token.restype = ctypes.c_int
lib.ck_true_bpe_add_special_token.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int32]

lib.ck_true_bpe_lookup.restype = ctypes.c_int32
lib.ck_true_bpe_lookup.argtypes = [ctypes.c_void_p, ctypes.c_char_p]

lib.ck_true_bpe_id_to_token.restype = ctypes.c_char_p
lib.ck_true_bpe_id_to_token.argtypes = [ctypes.c_void_p, ctypes.c_int32]

lib.ck_true_bpe_encode.restype = ctypes.c_int
lib.ck_true_bpe_encode.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int]

lib.ck_true_bpe_decode.restype = ctypes.c_int
lib.ck_true_bpe_decode.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_int]

lib.ck_true_bpe_detect_space_style.restype = ctypes.c_int
lib.ck_true_bpe_detect_space_style.argtypes = [ctypes.c_void_p]


class CKBPEConfig(ctypes.Structure):
    _fields_ = [
        ("add_bos", ctypes.c_bool),
        ("add_eos", ctypes.c_bool),
        ("byte_fallback", ctypes.c_bool),
        ("space_prefix_style", ctypes.c_int),
        ("pretokenizer", ctypes.c_int),
    ]


lib.ck_true_bpe_set_config.restype = None
lib.ck_true_bpe_set_config.argtypes = [ctypes.c_void_p, ctypes.POINTER(CKBPEConfig)]

lib.ck_true_bpe_vocab_size.restype = ctypes.c_size_t
lib.ck_true_bpe_vocab_size.argtypes = [ctypes.c_void_p]

CK_SPACE_PREFIX_SPM = 2
CK_SPACE_PREFIX_GPT2 = 1
CK_BPE_PRETOKENIZER_UNICODE_SPLIT_ISOLATED = 1
REQUIRE_HF_ORACLE = os.environ.get("CK_TOKENIZER_REQUIRE_HF_ORACLE") == "1"
_HF_TOKENIZER = None


def load_hf_qwen_tokenizer():
    """Load the independent oracle once; strict gates may not silently skip it."""
    global _HF_TOKENIZER
    if _HF_TOKENIZER is not None:
        return _HF_TOKENIZER
    if not HAS_HF:
        print("FAIL: transformers is required by the strict tokenizer parity gate")
        return None
    try:
        _HF_TOKENIZER = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
    except Exception as exc:
        print(f"FAIL: Could not load Qwen tokenizer oracle: {exc}")
        return None
    return _HF_TOKENIZER


# ═══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def encode_c(bpe, text: str) -> list:
    """Encode text using C tokenizer."""
    text_bytes = text.encode('utf-8')
    max_ids = len(text_bytes) * 4 + 100
    ids_array = (ctypes.c_int32 * max_ids)()
    num_ids = lib.ck_true_bpe_encode(bpe, text_bytes, len(text_bytes), ids_array, max_ids)
    return list(ids_array[:num_ids])


def decode_c(bpe, ids: list) -> str:
    """Decode token IDs using C tokenizer."""
    ids_array = (ctypes.c_int32 * len(ids))(*ids)
    out_buf = ctypes.create_string_buffer(len(ids) * 32)
    out_len = lib.ck_true_bpe_decode(bpe, ids_array, len(ids), out_buf, len(out_buf))
    return out_buf.value[:out_len].decode('utf-8', errors='replace')


def decode_c_bytes(bpe, ids: list) -> bytes:
    """Decode token IDs without applying a lossy UTF-8 error policy."""
    ids_array = (ctypes.c_int32 * len(ids))(*ids)
    out_buf = ctypes.create_string_buffer(max(32, len(ids) * 8))
    out_len = lib.ck_true_bpe_decode(bpe, ids_array, len(ids), out_buf, len(out_buf))
    return out_buf.raw[:out_len]


def gpt2_bytes_to_unicode() -> dict[int, str]:
    """Independent reference construction used by OpenAI GPT-2 tokenizers."""
    byte_values = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    codepoints = byte_values[:]
    mapped = 0
    for byte in range(256):
        if byte in byte_values:
            continue
        byte_values.append(byte)
        codepoints.append(256 + mapped)
        mapped += 1
    return dict(zip(byte_values, (chr(cp) for cp in codepoints)))


def load_vocab_from_hf(bpe, hf_tokenizer, *, load_merges: bool = False):
    """Load Hugging Face vocabulary and, when requested, ordered BPE merges."""
    vocab = hf_tokenizer.get_vocab()
    for token, id_val in vocab.items():
        lib.ck_true_bpe_add_token(bpe, token.encode('utf-8'), id_val, 0.0)

    if not load_merges:
        return

    backend = json.loads(hf_tokenizer.backend_tokenizer.to_str())
    for priority, pair in enumerate(backend.get("model", {}).get("merges", [])):
        if isinstance(pair, str):
            left, right = pair.split(" ", 1)
        else:
            left, right = pair
        left_id = vocab.get(left, -1)
        right_id = vocab.get(right, -1)
        merged_id = vocab.get(left + right, -1)
        if left_id >= 0 and right_id >= 0 and merged_id >= 0:
            lib.ck_true_bpe_add_merge(bpe, left_id, right_id, merged_id, priority)


def test_late_spm_space_prefix_detection():
    """Detect SentencePiece markers even when they appear past early vocab IDs."""
    print("\nTest: Late SentencePiece Space Prefix Detection")
    print("-" * 60)

    bpe = lib.ck_true_bpe_create()
    if not bpe:
        print("FAIL: Could not create tokenizer")
        return False

    try:
        for idx in range(10050):
            token = f"tok_{idx}".encode("utf-8")
            if lib.ck_true_bpe_add_token(bpe, token, idx, 0.0) != 0:
                print(f"FAIL: Could not add filler token {idx}")
                return False

        if lib.ck_true_bpe_add_token(bpe, "▁late".encode("utf-8"), 10050, 0.0) != 0:
            print("FAIL: Could not add late SPM token")
            return False

        style = int(lib.ck_true_bpe_detect_space_style(bpe))
        if style != CK_SPACE_PREFIX_SPM:
            print(f"FAIL: Expected SPM style ({CK_SPACE_PREFIX_SPM}), got {style}")
            return False

        decoded = decode_c(bpe, [10050])
        if decoded != " late":
            print(f"FAIL: Expected decoded text ' late', got {decoded!r}")
            return False

        print("PASS: Late SPM marker is detected and decoded as space")
        return True
    finally:
        lib.ck_true_bpe_free(bpe)


def test_unicode_isolated_pretokenizer_keeps_punctuation_out_of_word_merges():
    print("\nTest: Unicode Isolated Pretokenizer Boundaries")
    print("-" * 60)
    bpe = lib.ck_true_bpe_create()
    if not bpe:
        return False
    try:
        for token, token_id in (("_", 0), ("a", 1), ("_a", 2)):
            if lib.ck_true_bpe_add_token(bpe, token.encode("utf-8"), token_id, 0.0) != 0:
                return False
        if lib.ck_true_bpe_add_merge(bpe, 0, 1, 2, 0) != 0:
            return False
        cfg = CKBPEConfig(False, False, True, CK_SPACE_PREFIX_GPT2,
                          CK_BPE_PRETOKENIZER_UNICODE_SPLIT_ISOLATED)
        lib.ck_true_bpe_set_config(bpe, ctypes.byref(cfg))
        ids = encode_c(bpe, "_a")
        if ids != [0, 1]:
            print(f"FAIL: Expected isolated '_' and 'a', got {ids}")
            return False
        print("PASS: Punctuation cannot merge across the declared split boundary")
        return True
    finally:
        lib.ck_true_bpe_free(bpe)

# ═══════════════════════════════════════════════════════════════════════════════
# TEST: SPECIAL TOKEN ENCODING
# ═══════════════════════════════════════════════════════════════════════════════

def test_special_token_encoding():
    """Test that special tokens are encoded as single tokens, not broken into chars."""
    print("\n" + "="*60)
    print("TEST: Special Token Encoding")
    print("="*60)

    # Load Qwen tokenizer (has ChatML special tokens)
    print("\nLoading Qwen2-0.5B-Instruct tokenizer...")
    hf = load_hf_qwen_tokenizer()
    if hf is None:
        return not REQUIRE_HF_ORACLE

    # Create C tokenizer
    bpe = lib.ck_true_bpe_create()
    if not bpe:
        print("FAIL: Could not create C tokenizer")
        return False

    try:
        # Load vocab
        print("Loading vocabulary...")
        load_vocab_from_hf(bpe, hf)
        print(f"  Loaded {lib.ck_true_bpe_vocab_size(bpe)} tokens")

        # Register special tokens
        special_tokens = ["<|im_start|>", "<|im_end|>", "<|endoftext|>"]
        print("\nRegistering special tokens for pre-BPE matching...")
        for tok in special_tokens:
            tok_id = lib.ck_true_bpe_lookup(bpe, tok.encode('utf-8'))
            if tok_id >= 0:
                lib.ck_true_bpe_add_special_token(bpe, tok.encode('utf-8'), tok_id)
                print(f"  {tok} -> {tok_id}")

        # Test cases
        test_cases = [
            ("<|im_start|>system\nYou are helpful.<|im_end|>", "ChatML system message"),
            ("<|im_start|>user\nHello<|im_end|>", "ChatML user message"),
            ("Hello<|im_end|>world", "Special token in middle"),
            ("<|im_start|>", "Just special token"),
        ]

        print("\nTesting special token encoding:")
        print("-"*60)

        all_pass = True
        for text, desc in test_cases:
            # HuggingFace encoding
            hf_ids = hf.encode(text, add_special_tokens=False)

            # C encoding
            c_ids = encode_c(bpe, text)

            # Check if special tokens are encoded as single tokens
            match = hf_ids == c_ids

            # Also check that special tokens appear as single IDs
            im_start_id = hf.convert_tokens_to_ids("<|im_start|>")
            im_end_id = hf.convert_tokens_to_ids("<|im_end|>")

            # Count special token occurrences
            hf_special_count = hf_ids.count(im_start_id) + hf_ids.count(im_end_id)
            c_special_count = c_ids.count(im_start_id) + c_ids.count(im_end_id) if im_start_id else 0

            status = "PASS" if match else "FAIL"
            if not match:
                all_pass = False

            print(f"\n  {status}: {desc}")
            print(f"    Input: {text[:50]}...")
            print(f"    HF:  {hf_ids[:10]}... ({len(hf_ids)} tokens, {hf_special_count} special)")
            print(f"    C:   {c_ids[:10]}... ({len(c_ids)} tokens)")

            if not match:
                # Show diff
                for i, (h, c) in enumerate(zip(hf_ids, c_ids)):
                    if h != c:
                        h_tok = hf.decode([h])
                        c_tok = decode_c(bpe, [c])
                        print(f"    Diff at {i}: HF={h}({h_tok!r}) vs C={c}({c_tok!r})")
                        break

        return all_pass

    finally:
        lib.ck_true_bpe_free(bpe)


# ═══════════════════════════════════════════════════════════════════════════════
# TEST: GPT-2 BYTE DECODING
# ═══════════════════════════════════════════════════════════════════════════════

def test_gpt2_byte_decoding():
    """Test that GPT-2 byte-level encoded characters are decoded correctly."""
    print("\n" + "="*60)
    print("TEST: GPT-2 Byte Decoding")
    print("="*60)

    # Load Qwen tokenizer (uses GPT-2 style byte encoding)
    print("\nLoading Qwen2-0.5B-Instruct tokenizer...")
    hf = load_hf_qwen_tokenizer()
    if hf is None:
        return not REQUIRE_HF_ORACLE

    # Create C tokenizer
    bpe = lib.ck_true_bpe_create()
    if not bpe:
        print("FAIL: Could not create C tokenizer")
        return False

    try:
        # Load vocab
        print("Loading vocabulary...")
        load_vocab_from_hf(bpe, hf, load_merges=True)
        print(f"  Loaded {lib.ck_true_bpe_vocab_size(bpe)} tokens")

        # Test cases with special characters
        test_cases = [
            ("Hello\nWorld", "newline"),
            ("Tab\there", "tab character"),
            ("Line1\n\nLine2", "double newline"),
            ("Hello World", "regular space"),
            ("Path/to/file", "slash"),
            ("café naïve résumé", "accented Latin"),
            ("日本語 中文 한국어", "multilingual CJK"),
            ("Привет мир Ελληνικά", "Cyrillic and Greek"),
            ("مرحبا بالعالم नमस्ते", "Arabic and Devanagari"),
            ("e\u0301 != é", "combining character"),
            ("🎉🚀👍🏽", "emoji and skin tone"),
            ("👨‍👩‍👧‍👦 🇨🇦", "joined emoji and flag"),
            ("  leading\tspace\ntrailing  ", "mixed whitespace"),
            ("“quotes”—€12.50…", "Unicode punctuation and currency"),
            ('printf("%s\\n", name); // π≈3.14159', "code and symbols"),
        ]

        print("\nTesting GPT-2 byte decoding:")
        print("-"*60)

        all_pass = True
        for text, desc in test_cases:
            # Encode then decode with HuggingFace
            hf_ids = hf.encode(text, add_special_tokens=False)
            hf_decoded = hf.decode(hf_ids)

            # C encoding must select the same tokens. A C-only round trip can
            # hide a symmetrically incorrect byte map.
            c_ids = encode_c(bpe, unicodedata.normalize("NFC", text))

            # Decode with C tokenizer (using same IDs from HF)
            c_decoded = decode_c(bpe, hf_ids)

            # Compare
            match = c_ids == hf_ids and c_decoded == hf_decoded
            status = "PASS" if match else "FAIL"
            if not match:
                all_pass = False

            print(f"\n  {status}: {desc}")
            print(f"    Original: {text!r}")
            print(f"    IDs: {hf_ids}")
            print(f"    C IDs match: {c_ids == hf_ids}")
            print(f"    HF decode:  {hf_decoded!r}")
            print(f"    C decode:   {c_decoded!r}")

        return all_pass

    finally:
        lib.ck_true_bpe_free(bpe)


def test_exhaustive_gpt2_byte_table():
    """Every GPT-2 mapped codepoint must recover its original byte exactly."""
    print("\n" + "="*60)
    print("TEST: Exhaustive GPT-2 Byte Table")
    print("="*60)

    bpe = lib.ck_true_bpe_create()
    if not bpe:
        print("FAIL: Could not create C tokenizer")
        return False

    try:
        byte_map = gpt2_bytes_to_unicode()
        for byte in range(256):
            token = byte_map[byte].encode("utf-8")
            if lib.ck_true_bpe_add_token(bpe, token, byte, 0.0) != 0:
                print(f"FAIL: Could not add byte token {byte}")
                return False

        decoded = decode_c_bytes(bpe, list(range(256)))
        expected = bytes(range(256))
        if decoded != expected:
            first = next(i for i, (actual, wanted) in enumerate(zip(decoded, expected)) if actual != wanted)
            print(f"FAIL: byte {first:#04x} decoded as {decoded[first]:#04x}")
            return False

        print("PASS: all 256 GPT-2 byte mappings decode exactly")
        return True
    finally:
        lib.ck_true_bpe_free(bpe)


# ═══════════════════════════════════════════════════════════════════════════════
# TEST: CHAT TEMPLATE
# ═══════════════════════════════════════════════════════════════════════════════

def test_chat_template_encoding():
    """Test that full chat templates encode correctly."""
    print("\n" + "="*60)
    print("TEST: Chat Template Encoding")
    print("="*60)

    # Load Qwen tokenizer
    print("\nLoading Qwen2-0.5B-Instruct tokenizer...")
    hf = load_hf_qwen_tokenizer()
    if hf is None:
        return not REQUIRE_HF_ORACLE

    # Create C tokenizer
    bpe = lib.ck_true_bpe_create()
    if not bpe:
        print("FAIL: Could not create C tokenizer")
        return False

    try:
        # Load vocab
        print("Loading vocabulary...")
        load_vocab_from_hf(bpe, hf, load_merges=True)

        # Register special tokens
        special_tokens = ["<|im_start|>", "<|im_end|>", "<|endoftext|>"]
        for tok in special_tokens:
            tok_id = lib.ck_true_bpe_lookup(bpe, tok.encode('utf-8'))
            if tok_id >= 0:
                lib.ck_true_bpe_add_special_token(bpe, tok.encode('utf-8'), tok_id)

        # Full ChatML prompt
        prompt = """<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
Hello, how are you?<|im_end|>
<|im_start|>assistant
"""

        print("\nTesting full chat template:")
        print("-"*60)
        print(f"Prompt:\n{prompt}")

        # HuggingFace encoding
        hf_ids = hf.encode(prompt, add_special_tokens=False)

        # C encoding
        c_ids = encode_c(bpe, prompt)

        print(f"\nHF IDs ({len(hf_ids)} tokens): {hf_ids[:20]}...")
        print(f"C  IDs ({len(c_ids)} tokens):  {c_ids[:20]}...")

        # Check for key special tokens in output
        im_start_id = hf.convert_tokens_to_ids("<|im_start|>")
        im_end_id = hf.convert_tokens_to_ids("<|im_end|>")

        hf_starts = hf_ids.count(im_start_id)
        hf_ends = hf_ids.count(im_end_id)
        c_starts = c_ids.count(im_start_id)
        c_ends = c_ids.count(im_end_id)

        print(f"\nSpecial token counts:")
        print(f"  <|im_start|> ({im_start_id}): HF={hf_starts}, C={c_starts}")
        print(f"  <|im_end|> ({im_end_id}): HF={hf_ends}, C={c_ends}")

        # Check parity
        match = hf_ids == c_ids
        if match:
            print("\nPASS: Full parity with HuggingFace!")
        else:
            print("\nFAIL: Token mismatch")
            # Find first diff
            for i, (h, c) in enumerate(zip(hf_ids, c_ids)):
                if h != c:
                    h_tok = hf.decode([h])
                    c_tok = decode_c(bpe, [c])
                    print(f"  First diff at position {i}: HF={h}({h_tok!r}) vs C={c}({c_tok!r})")
                    break

        return match

    finally:
        lib.ck_true_bpe_free(bpe)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("="*60)
    print("True BPE Tokenizer Test: Special Tokens & Byte Decoding")
    print("="*60)

    results = []

    # Run tests
    results.append(("Special Token Encoding", test_special_token_encoding()))
    results.append(("GPT-2 Byte Decoding", test_gpt2_byte_decoding()))
    results.append(("Exhaustive GPT-2 Byte Table", test_exhaustive_gpt2_byte_table()))
    results.append(("Late SPM Space Prefix Detection", test_late_spm_space_prefix_detection()))
    results.append(("Unicode Isolated Pretokenizer", test_unicode_isolated_pretokenizer_keeps_punctuation_out_of_word_merges()))
    results.append(("Chat Template Encoding", test_chat_template_encoding()))

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    all_pass = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {status}: {name}")
        if not passed:
            all_pass = False

    print("="*60)

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())

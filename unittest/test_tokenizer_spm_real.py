#!/usr/bin/env python3
"""
Real SPM Parity Tests - Compares C tokenizer against Python SentencePiece

This test:
1. Creates a real SPM model with SentencePiece
2. Extracts vocab/scores/types
3. Loads into C tokenizer via ck_tokenizer_load_binary_with_scores
4. Compares tokenization output

Usage: python test_tokenizer_spm_real.py [--verbose]
"""

import argparse
import array
import ctypes
import os
import struct
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional, Tuple

# ANSI colors
RED = '\033[91m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
CYAN = '\033[96m'
RESET = '\033[0m'
BOLD = '\033[1m'


def main():
    parser = argparse.ArgumentParser(description="Real SPM Parity Tests")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()

    print(f"\n{BOLD}{'='*80}{RESET}")
    print(f"{BOLD}  REAL SPM PARITY TESTS{RESET}")
    print(f"{BOLD}{'='*80}{RESET}\n")

    # Load C library
    lib_path = None
    for name in ["libckernel_tokenizer.so", "libckernel_engine.so"]:
        for path in [Path.cwd() / name, Path.cwd() / "build" / name]:
            if path.exists():
                lib_path = path
                break
        if lib_path:
            break

    if not lib_path:
        print(f"{RED}ERROR: Could not find tokenizer library{RESET}")
        print("Run 'make tokenizer' first")
        return 1

    print(f"{CYAN}[INFO]{RESET} Loading library: {lib_path}")
    lib = ctypes.CDLL(str(lib_path))

    # Setup function signatures
    lib.ck_tokenizer_create.restype = ctypes.c_void_p
    lib.ck_tokenizer_create.argtypes = [ctypes.c_int]
    lib.ck_tokenizer_free.restype = None
    lib.ck_tokenizer_free.argtypes = [ctypes.c_void_p]
    lib.ck_tokenizer_encode.restype = ctypes.c_int
    lib.ck_tokenizer_encode.argtypes = [
        ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int,
        ctypes.POINTER(ctypes.c_int32), ctypes.c_int
    ]
    lib.ck_tokenizer_load_binary_with_scores.restype = ctypes.c_int
    lib.ck_tokenizer_load_binary_with_scores.argtypes = [
        ctypes.c_void_p, ctypes.c_int,
        ctypes.POINTER(ctypes.c_int32), ctypes.c_char_p,
        ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_int, ctypes.c_void_p
    ]
    lib.ck_tokenizer_set_special_ids.restype = None
    lib.ck_tokenizer_set_special_ids.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32
    ]
    lib.ck_tokenizer_set_add_bos_eos.restype = None
    lib.ck_tokenizer_set_add_bos_eos.argtypes = [ctypes.c_void_p, ctypes.c_bool, ctypes.c_bool]
    lib.ck_tokenizer_set_add_space_prefix.restype = None
    lib.ck_tokenizer_set_add_space_prefix.argtypes = [ctypes.c_void_p, ctypes.c_bool]

    # Try to load SentencePiece
    try:
        import sentencepiece as spm
        print(f"{GREEN}[OK]{RESET} sentencepiece available")
    except ImportError:
        print(f"{RED}ERROR: sentencepiece not installed{RESET}")
        print(f"{YELLOW}Install with: pip install sentencepiece{RESET}")
        return 1

    # Create SPM models
    print(f"\n{CYAN}[STEP 1]{RESET} Creating SPM models...")

    train_text = """Hello world this is a test.
Sentence piece tokenizer handles spaces differently.
The quick brown fox jumps over the lazy dog.
Unicode chars like café and 你好 should work.
Multiple   spaces   should   be   handled.
Special chars: @#$%^&*()[]{}|\\/:;\"'<>?,.`~
"""

    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(train_text * 20)  # Much more repetitions
        train_file = f.name

    def train_spm_model(add_dummy_prefix=True, bos_id=-1, eos_id=-1):
        model_prefix = tempfile.mktemp()
        spm.SentencePieceTrainer.train(
            input=train_file,
            model_prefix=model_prefix,
            vocab_size=80,  # Small vocab for small corpus
            model_type="unigram",
            pad_id=-1,
            bos_id=bos_id,
            eos_id=eos_id,
            unk_id=0,
            add_dummy_prefix=add_dummy_prefix,
        )
        sp = spm.SentencePieceProcessor()
        sp.load(f"{model_prefix}.model")
        return sp, model_prefix

    def create_ck_tokenizer_from_sp(sp, add_space_prefix, add_bos, add_eos):
        # Extract vocab data for C loader
        vocab_size = sp.get_piece_size()
        offsets = []
        strings_parts = []
        scores = []
        types = []
        offset = 0

        for i in range(vocab_size):
            piece = sp.id_to_piece(i)
            score = sp.get_score(i)

            offsets.append(offset)
            strings_parts.append(piece)
            # Use byte length (utf-8 encoded), not character count
            offset += len(piece.encode('utf-8')) + 1
            scores.append(score)

            # Map SentencePiece token types to GGUF token types
            # GGUF: NORMAL=1, UNKNOWN=2, CONTROL=3, USER_DEFINED=4, UNUSED=5, BYTE=6
            try:
                if hasattr(sp, "is_unknown") and sp.is_unknown(i):
                    types.append(2)
                elif hasattr(sp, "is_control") and sp.is_control(i):
                    types.append(3)
                elif hasattr(sp, "is_user_defined") and sp.is_user_defined(i):
                    types.append(4)
                elif hasattr(sp, "is_unused") and sp.is_unused(i):
                    types.append(5)
                elif hasattr(sp, "is_byte") and sp.is_byte(i):
                    types.append(6)
                else:
                    types.append(1)
            except Exception:
                types.append(1)

        # Build string pool
        strings_pool = b'\x00'.join(p.encode('utf-8') for p in strings_parts) + b'\x00'

        tok = lib.ck_tokenizer_create(2)  # CK_TOKENIZER_SPM = 2
        if not tok:
            raise RuntimeError("Failed to create tokenizer")

        offsets_arr = (ctypes.c_int32 * vocab_size)(*offsets)
        scores_arr = (ctypes.c_float * vocab_size)(*scores)
        types_arr = (ctypes.c_uint8 * vocab_size)(*types)

        result = lib.ck_tokenizer_load_binary_with_scores(
            tok, vocab_size,
            offsets_arr, strings_pool,
            scores_arr, types_arr,
            0, None
        )
        if result != 0:
            lib.ck_tokenizer_free(tok)
            raise RuntimeError(f"Failed to load vocab (error {result})")

        # Apply config knobs that differ by model
        lib.ck_tokenizer_set_add_space_prefix(tok, add_space_prefix)
        lib.ck_tokenizer_set_add_bos_eos(tok, add_bos, add_eos)
        lib.ck_tokenizer_set_special_ids(
            tok,
            sp.unk_id(),
            sp.bos_id(),
            sp.eos_id(),
            sp.pad_id(),
            -1
        )
        return tok

    def encode_c(tok, text):
        text_bytes = text.encode('utf-8')
        ids = (ctypes.c_int32 * 256)()
        num = lib.ck_tokenizer_encode(tok, text_bytes, len(text_bytes), ids, 256)
        return list(ids[:num])

    passed = 0
    failed = 0

    def run_suite(sp, tok, test_cases, add_bos, add_eos, title, include_byte_test=False):
        nonlocal passed, failed
        print(f"\n{CYAN}[STEP 2]{RESET} {title}")

        for text, desc in test_cases:
            start = time.perf_counter()
            c_ids = encode_c(tok, text)
            c_time = (time.perf_counter() - start) * 1000

            spm_ids = sp.encode_as_ids(text)
            if add_bos and sp.bos_id() >= 0:
                spm_ids = [sp.bos_id()] + spm_ids
            if add_eos and sp.eos_id() >= 0:
                spm_ids = spm_ids + [sp.eos_id()]

            match = c_ids == spm_ids
            match_str = f"{GREEN}MATCH{RESET}" if match else f"{RED}MISMATCH{RESET}"
            print(f"  {desc:<30} C={len(c_ids):2d} tokens, SPM={len(spm_ids):2d} tokens  {match_str}")

            if args.verbose:
                print(f"    C tokens:    {c_ids}")
                print(f"    SPM tokens: {spm_ids}")

            if match:
                passed += 1
            else:
                failed += 1

        if include_byte_test:
            byte_text = "\x00\x01"
            try:
                c_ids = encode_c(tok, byte_text)
                spm_ids = sp.encode_as_ids(byte_text)
                if add_bos and sp.bos_id() >= 0:
                    spm_ids = [sp.bos_id()] + spm_ids
                if add_eos and sp.eos_id() >= 0:
                    spm_ids = spm_ids + [sp.eos_id()]
                match = c_ids == spm_ids
                match_str = f"{GREEN}MATCH{RESET}" if match else f"{RED}MISMATCH{RESET}"
                print(f"  {'Byte/control chars':<30} C={len(c_ids):2d} tokens, SPM={len(spm_ids):2d} tokens  {match_str}")
                if match:
                    passed += 1
                else:
                    failed += 1
            except Exception:
                print(f"  {YELLOW}Byte/control chars test skipped (SentencePiece rejected input){RESET}")

    # Test cases
    test_cases = [
        ("Hello world", "Basic ASCII with spaces"),
        ("  leading space", "Leading spaces"),
        ("trailing space ", "Trailing spaces"),
        ("multiple   spaces", "Multiple spaces"),
        ("café", "Unicode with accent"),
        ("你好", "Chinese characters"),
        ("Hello你好world", "Mixed scripts"),
        ("a", "Single character"),
        ("", "Empty string"),
        ("special chars @#$%", "Special characters"),
        ("123 456 789", "Numbers with spaces"),
    ]

    # Default SPM (add_dummy_prefix=true, no BOS/EOS)
    sp_default, model_prefix_default = train_spm_model(add_dummy_prefix=True, bos_id=-1, eos_id=-1)
    print(f"{GREEN}[OK]{RESET} Loaded default SPM model with {sp_default.get_piece_size()} tokens")
    tok_default = create_ck_tokenizer_from_sp(sp_default, add_space_prefix=True, add_bos=False, add_eos=False)
    run_suite(sp_default, tok_default, test_cases, add_bos=False, add_eos=False,
              title="Default SPM (add_dummy_prefix=true)", include_byte_test=True)
    lib.ck_tokenizer_free(tok_default)

    # SPM with add_dummy_prefix=false (should NOT add ▁ prefix)
    sp_noprefix, model_prefix_noprefix = train_spm_model(add_dummy_prefix=False, bos_id=-1, eos_id=-1)
    print(f"{GREEN}[OK]{RESET} Loaded SPM model (add_dummy_prefix=false) with {sp_noprefix.get_piece_size()} tokens")
    tok_noprefix = create_ck_tokenizer_from_sp(sp_noprefix, add_space_prefix=False, add_bos=False, add_eos=False)
    run_suite(sp_noprefix, tok_noprefix, test_cases, add_bos=False, add_eos=False,
              title="SPM add_dummy_prefix=false")
    lib.ck_tokenizer_free(tok_noprefix)

    # SPM with BOS/EOS enabled
    sp_bos, model_prefix_bos = train_spm_model(add_dummy_prefix=True, bos_id=1, eos_id=2)
    print(f"{GREEN}[OK]{RESET} Loaded SPM model (BOS/EOS) with {sp_bos.get_piece_size()} tokens")
    tok_bos = create_ck_tokenizer_from_sp(sp_bos, add_space_prefix=True, add_bos=True, add_eos=True)
    run_suite(sp_bos, tok_bos, test_cases, add_bos=True, add_eos=True,
              title="SPM with BOS/EOS")
    lib.ck_tokenizer_free(tok_bos)

    # Cleanup model files
    for prefix in [model_prefix_default, model_prefix_noprefix, model_prefix_bos]:
        for ext in ['.model', '.vocab']:
            try:
                os.unlink(f"{prefix}{ext}")
            except Exception:
                pass

    # Cleanup train file
    try:
        os.unlink(train_file)
    except Exception:
        pass

    # Summary
    print(f"\n{BOLD}{'='*80}{RESET}")
    print(f"{BOLD}  SUMMARY{RESET}")
    print(f"{BOLD}{'='*80}{RESET}")
    print(f"\n  {GREEN}Passed: {passed}{RESET}")
    print(f"  {RED}Failed: {failed}{RESET}")

    if failed == 0:
        print(f"\n{GREEN}All parity tests passed!{RESET}")
        return 0
    else:
        print(f"\n{RED}Some parity tests failed!{RESET}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

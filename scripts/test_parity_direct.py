#!/usr/bin/env python3
"""
Direct parity test: Compare CK engine vs llama.cpp logits

This script:
1. Loads both models via ctypes
2. Runs a single token forward pass
3. Compares the output logits
4. Reports the divergence

Usage:
    python scripts/test_parity_direct.py --gguf model.gguf --model-dir /path/to/ck/model
"""

import ctypes
import numpy as np
import argparse
import json
from pathlib import Path
from typing import Optional, Tuple

# =============================================================================
# LLAMA.CPP BINDINGS
# =============================================================================

class LlamaCpp:
    """Wrapper for llama.cpp libllama.so"""

    def __init__(self, lib_dir: str):
        lib_path = Path(lib_dir)

        # Load libraries in dependency order
        self.libggml_base = ctypes.CDLL(str(lib_path / "libggml-base.so"), mode=ctypes.RTLD_GLOBAL)
        self.libggml_cpu = ctypes.CDLL(str(lib_path / "libggml-cpu.so"), mode=ctypes.RTLD_GLOBAL)
        self.libggml = ctypes.CDLL(str(lib_path / "libggml.so"), mode=ctypes.RTLD_GLOBAL)
        self.libllama = ctypes.CDLL(str(lib_path / "libllama.so"))

        # Define structures
        class llama_model_params(ctypes.Structure):
            _fields_ = [
                ("devices", ctypes.c_void_p),
                ("tensor_split", ctypes.c_void_p),
                ("n_gpu_layers", ctypes.c_int),
                ("split_mode", ctypes.c_int),
                ("main_gpu", ctypes.c_int),
                ("vocab_only", ctypes.c_bool),
                ("use_mmap", ctypes.c_bool),
                ("use_mlock", ctypes.c_bool),
                ("check_tensors", ctypes.c_bool),
            ]

        class llama_context_params(ctypes.Structure):
            _fields_ = [
                ("n_ctx", ctypes.c_uint32),
                ("n_batch", ctypes.c_uint32),
                ("n_ubatch", ctypes.c_uint32),
                ("n_seq_max", ctypes.c_uint32),
                ("n_threads", ctypes.c_int32),
                ("n_threads_batch", ctypes.c_int32),
                ("rope_scaling_type", ctypes.c_int32),
                ("pooling_type", ctypes.c_int32),
                ("attention_type", ctypes.c_int32),
                ("rope_freq_base", ctypes.c_float),
                ("rope_freq_scale", ctypes.c_float),
                ("yarn_ext_factor", ctypes.c_float),
                ("yarn_attn_factor", ctypes.c_float),
                ("yarn_beta_fast", ctypes.c_float),
                ("yarn_beta_slow", ctypes.c_float),
                ("yarn_orig_ctx", ctypes.c_uint32),
                ("defrag_thold", ctypes.c_float),
                ("cb_eval", ctypes.c_void_p),
                ("cb_eval_user_data", ctypes.c_void_p),
                ("type_k", ctypes.c_int32),
                ("type_v", ctypes.c_int32),
                ("logits_all", ctypes.c_bool),
                ("embeddings", ctypes.c_bool),
                ("offload_kqv", ctypes.c_bool),
                ("flash_attn", ctypes.c_bool),
                ("no_perf", ctypes.c_bool),
                ("abort_callback", ctypes.c_void_p),
                ("abort_callback_data", ctypes.c_void_p),
            ]

        class llama_batch(ctypes.Structure):
            _fields_ = [
                ("n_tokens", ctypes.c_int32),
                ("token", ctypes.POINTER(ctypes.c_int32)),
                ("embd", ctypes.c_void_p),
                ("pos", ctypes.POINTER(ctypes.c_int32)),
                ("n_seq_id", ctypes.POINTER(ctypes.c_int32)),
                ("seq_id", ctypes.POINTER(ctypes.POINTER(ctypes.c_int32))),
                ("logits", ctypes.POINTER(ctypes.c_int8)),
            ]

        self.llama_model_params = llama_model_params
        self.llama_context_params = llama_context_params
        self.llama_batch = llama_batch

        # Set up function signatures
        self.libllama.llama_model_default_params.argtypes = []
        self.libllama.llama_model_default_params.restype = llama_model_params

        self.libllama.llama_context_default_params.argtypes = []
        self.libllama.llama_context_default_params.restype = llama_context_params

        self.libllama.llama_model_load_from_file.argtypes = [ctypes.c_char_p, llama_model_params]
        self.libllama.llama_model_load_from_file.restype = ctypes.c_void_p

        self.libllama.llama_init_from_model.argtypes = [ctypes.c_void_p, llama_context_params]
        self.libllama.llama_init_from_model.restype = ctypes.c_void_p

        self.libllama.llama_model_n_embd.argtypes = [ctypes.c_void_p]
        self.libllama.llama_model_n_embd.restype = ctypes.c_int32

        self.libllama.llama_model_n_layer.argtypes = [ctypes.c_void_p]
        self.libllama.llama_model_n_layer.restype = ctypes.c_int32

        self.libllama.llama_vocab_n_tokens.argtypes = [ctypes.c_void_p]
        self.libllama.llama_vocab_n_tokens.restype = ctypes.c_int32

        self.libllama.llama_model_get_vocab.argtypes = [ctypes.c_void_p]
        self.libllama.llama_model_get_vocab.restype = ctypes.c_void_p

        self.libllama.llama_decode.argtypes = [ctypes.c_void_p, llama_batch]
        self.libllama.llama_decode.restype = ctypes.c_int32

        self.libllama.llama_get_logits.argtypes = [ctypes.c_void_p]
        self.libllama.llama_get_logits.restype = ctypes.POINTER(ctypes.c_float)

        self.libllama.llama_batch_init.argtypes = [ctypes.c_int32, ctypes.c_int32, ctypes.c_int32]
        self.libllama.llama_batch_init.restype = llama_batch

        self.libllama.llama_batch_free.argtypes = [llama_batch]
        self.libllama.llama_batch_free.restype = None

        self.libllama.llama_model_free.argtypes = [ctypes.c_void_p]
        self.libllama.llama_model_free.restype = None

        self.libllama.llama_free.argtypes = [ctypes.c_void_p]
        self.libllama.llama_free.restype = None

        self.model = None
        self.ctx = None
        self.vocab_size = 0
        self.n_embd = 0
        self.n_layer = 0

    def load(self, gguf_path: str) -> bool:
        """Load model from GGUF file."""
        print(f"[LLAMA] Loading {gguf_path}...")

        # Get default params
        mparams = self.libllama.llama_model_default_params()
        mparams.use_mmap = True
        mparams.use_mlock = False
        mparams.n_gpu_layers = 0  # CPU only

        # Load model
        self.model = self.libllama.llama_model_load_from_file(
            gguf_path.encode(), mparams
        )
        if not self.model:
            print("[LLAMA] Failed to load model")
            return False

        # Get vocab
        vocab = self.libllama.llama_model_get_vocab(self.model)
        self.vocab_size = self.libllama.llama_vocab_n_tokens(vocab)
        self.n_embd = self.libllama.llama_model_n_embd(self.model)
        self.n_layer = self.libllama.llama_model_n_layer(self.model)

        print(f"[LLAMA] Loaded: vocab={self.vocab_size}, embd={self.n_embd}, layers={self.n_layer}")

        # Create context
        cparams = self.libllama.llama_context_default_params()
        cparams.n_ctx = 512
        cparams.n_batch = 512
        cparams.n_threads = 4
        cparams.flash_attn = False

        self.ctx = self.libllama.llama_init_from_model(self.model, cparams)
        if not self.ctx:
            print("[LLAMA] Failed to create context")
            return False

        print("[LLAMA] Context created")
        return True

    def forward(self, tokens: list) -> np.ndarray:
        """Run forward pass and return logits."""
        n_tokens = len(tokens)

        # Create batch
        batch = self.libllama.llama_batch_init(n_tokens, 0, 1)
        batch.n_tokens = n_tokens

        # Fill tokens
        for i, tok in enumerate(tokens):
            batch.token[i] = tok
            batch.pos[i] = i
            batch.n_seq_id[i] = 1
            seq_id_arr = (ctypes.c_int32 * 1)(0)
            batch.seq_id[i] = ctypes.cast(seq_id_arr, ctypes.POINTER(ctypes.c_int32))
            batch.logits[i] = 1 if i == n_tokens - 1 else 0

        # Run decode
        ret = self.libllama.llama_decode(self.ctx, batch)
        if ret != 0:
            print(f"[LLAMA] Decode failed: {ret}")
            return None

        # Get logits
        logits_ptr = self.libllama.llama_get_logits(self.ctx)
        logits = np.ctypeslib.as_array(logits_ptr, shape=(self.vocab_size,)).copy()

        self.libllama.llama_batch_free(batch)
        return logits

    def free(self):
        if self.ctx:
            self.libllama.llama_free(self.ctx)
        if self.model:
            self.libllama.llama_model_free(self.model)

# =============================================================================
# CK ENGINE BINDINGS
# =============================================================================

class CKEngine:
    """Wrapper for CK Engine libmodel.so"""

    def __init__(self, model_dir: str):
        self.model_dir = Path(model_dir)
        self.lib = None
        self.vocab_size = 0

    def load(self, weights_path: str = None) -> bool:
        """Load CK model."""
        lib_path = self.model_dir / "libmodel.so"
        if not lib_path.exists():
            print(f"[CK] Library not found: {lib_path}")
            return False

        print(f"[CK] Loading {lib_path}...")
        self.lib = ctypes.CDLL(str(lib_path))

        # Set up function signatures
        self.lib.ck_model_init.argtypes = [ctypes.c_char_p]
        self.lib.ck_model_init.restype = ctypes.c_int

        self.lib.ck_model_get_vocab_size.argtypes = []
        self.lib.ck_model_get_vocab_size.restype = ctypes.c_int

        self.lib.ck_model_embed_tokens.argtypes = [ctypes.POINTER(ctypes.c_int32), ctypes.c_int]
        self.lib.ck_model_embed_tokens.restype = ctypes.c_int

        self.lib.ck_model_forward.argtypes = [ctypes.POINTER(ctypes.c_float)]
        self.lib.ck_model_forward.restype = ctypes.c_int

        self.lib.ck_model_get_logits.argtypes = []
        self.lib.ck_model_get_logits.restype = ctypes.POINTER(ctypes.c_float)

        # Use default weights path if not specified
        if weights_path is None:
            weights_path = str(self.model_dir / "weights.bump")

        ret = self.lib.ck_model_init(weights_path.encode())
        if ret != 0:
            print(f"[CK] Failed to initialize: {ret}")
            return False

        self.vocab_size = self.lib.ck_model_get_vocab_size()
        print(f"[CK] Loaded: vocab={self.vocab_size}")
        return True

    def forward(self, tokens: list) -> np.ndarray:
        """Run forward pass and return logits."""
        n_tokens = len(tokens)

        # Create token array
        token_arr = (ctypes.c_int32 * n_tokens)(*tokens)

        # Embed tokens
        ret = self.lib.ck_model_embed_tokens(token_arr, n_tokens)
        if ret != 0:
            print(f"[CK] Embed failed: {ret}")
            return None

        # Run forward
        ret = self.lib.ck_model_forward(None)
        if ret != 0:
            print(f"[CK] Forward failed: {ret}")
            return None

        # Get logits
        logits_ptr = self.lib.ck_model_get_logits()
        # Last token's logits
        offset = (n_tokens - 1) * self.vocab_size
        logits = np.ctypeslib.as_array(logits_ptr, shape=((n_tokens) * self.vocab_size,))[offset:offset+self.vocab_size].copy()

        return logits

# =============================================================================
# PARITY TEST
# =============================================================================

def compare_logits(llama_logits: np.ndarray, ck_logits: np.ndarray) -> dict:
    """Compare two logit arrays and return statistics."""
    diff = np.abs(llama_logits - ck_logits)

    max_diff = np.max(diff)
    max_idx = np.argmax(diff)
    mean_diff = np.mean(diff)

    # Relative error (for non-zero values)
    rel_err = diff / (np.abs(llama_logits) + 1e-9)
    max_rel_err = np.max(rel_err)
    mean_rel_err = np.mean(rel_err)

    # Top-K comparison
    llama_top5 = np.argsort(llama_logits)[-5:][::-1]
    ck_top5 = np.argsort(ck_logits)[-5:][::-1]
    top5_match = np.array_equal(llama_top5, ck_top5)
    top1_match = llama_top5[0] == ck_top5[0]

    return {
        "max_diff": float(max_diff),
        "max_diff_idx": int(max_idx),
        "mean_diff": float(mean_diff),
        "max_rel_err": float(max_rel_err),
        "mean_rel_err": float(mean_rel_err),
        "llama_top5": llama_top5.tolist(),
        "ck_top5": ck_top5.tolist(),
        "top1_match": top1_match,
        "top5_match": top5_match,
        "llama_max": float(np.max(llama_logits)),
        "llama_argmax": int(np.argmax(llama_logits)),
        "ck_max": float(np.max(ck_logits)),
        "ck_argmax": int(np.argmax(ck_logits)),
    }

def main():
    parser = argparse.ArgumentParser(description="Direct parity test: CK vs llama.cpp")
    parser.add_argument("--gguf", required=True, help="Path to GGUF model file")
    parser.add_argument("--model-dir", required=True, help="Path to CK model directory")
    parser.add_argument("--token", type=int, default=1, help="Token ID to test (default: 1)")
    parser.add_argument("--llama-lib", default="llama.cpp/build/bin", help="Path to llama.cpp libraries")
    args = parser.parse_args()

    print("=" * 60)
    print("DIRECT PARITY TEST: CK Engine vs llama.cpp")
    print("=" * 60)

    # Load llama.cpp
    llama = LlamaCpp(args.llama_lib)
    if not llama.load(args.gguf):
        return 1

    # Load CK
    ck = CKEngine(args.model_dir)
    if not ck.load():
        return 1

    # Test single token
    test_token = args.token
    tokens = [test_token]

    print(f"\nTesting single token: {test_token}")
    print("-" * 40)

    # Run llama.cpp
    print("[LLAMA] Running forward pass...")
    llama_logits = llama.forward(tokens)
    if llama_logits is None:
        print("LLAMA forward failed!")
        return 1

    # Run CK
    print("[CK] Running forward pass...")
    ck_logits = ck.forward(tokens)
    if ck_logits is None:
        print("CK forward failed!")
        return 1

    # Compare
    print("\n" + "=" * 60)
    print("COMPARISON RESULTS")
    print("=" * 60)

    stats = compare_logits(llama_logits, ck_logits)

    print(f"\nMax absolute diff: {stats['max_diff']:.6f} at index {stats['max_diff_idx']}")
    print(f"Mean absolute diff: {stats['mean_diff']:.6f}")
    print(f"Max relative error: {stats['max_rel_err']:.2%}")
    print(f"Mean relative error: {stats['mean_rel_err']:.2%}")

    print(f"\nLLAMA top-5 tokens: {stats['llama_top5']}")
    print(f"CK top-5 tokens:    {stats['ck_top5']}")
    print(f"Top-1 match: {stats['top1_match']}")
    print(f"Top-5 match: {stats['top5_match']}")

    print(f"\nLLAMA argmax: {stats['llama_argmax']} (logit={stats['llama_max']:.4f})")
    print(f"CK argmax:    {stats['ck_argmax']} (logit={stats['ck_max']:.4f})")

    # Determine pass/fail
    passed = stats['top1_match'] and stats['max_rel_err'] < 0.1  # 10% tolerance

    print("\n" + "=" * 60)
    if passed:
        print("RESULT: PASS - Logits match within tolerance")
    else:
        print("RESULT: FAIL - Logits diverge significantly")
        print("\nPossible causes:")
        print("  1. Weight loading issue")
        print("  2. Kernel implementation bug")
        print("  3. Different RoPE implementation")
        print("  4. Attention implementation difference")
    print("=" * 60)

    # Cleanup
    llama.free()

    return 0 if passed else 1

if __name__ == "__main__":
    exit(main())

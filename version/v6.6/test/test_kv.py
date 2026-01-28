#!/usr/bin/env python3
"""Debug KV cache behavior."""
import ctypes
import numpy as np
from pathlib import Path

model_dir = Path.home() / '.cache/ck-engine-v6.6/models/Qwen--Qwen2-0.5B-Instruct-GGUF'
kernel_lib = ctypes.CDLL(str(model_dir / 'libckernel_engine.so'), mode=ctypes.RTLD_GLOBAL)
lib = ctypes.CDLL(str(model_dir / 'libmodel.so'))

lib.ck_model_init.argtypes = [ctypes.c_char_p]
lib.ck_model_init.restype = ctypes.c_int
lib.ck_model_decode.argtypes = [ctypes.c_int32, ctypes.POINTER(ctypes.c_float)]
lib.ck_model_decode.restype = ctypes.c_int
lib.ck_model_get_vocab_size.argtypes = []
lib.ck_model_get_vocab_size.restype = ctypes.c_int
lib.ck_model_kv_cache_reset.argtypes = []
lib.ck_model_kv_cache_reset.restype = None
lib.ck_model_free.argtypes = []
lib.ck_model_free.restype = None

weights_path = model_dir / 'weights.bump'
lib.ck_model_init(str(weights_path).encode())
lib.ck_model_kv_cache_reset()

vocab_size = lib.ck_model_get_vocab_size()
out = (ctypes.c_float * vocab_size)()

# Simple test: just token 9707 ("Hello")
# Without any prior context
print("Test 1: Single token decode")
lib.ck_model_decode(9707, out)
logits = np.array(out)
print(f"  Token 9707 -> argmax: {np.argmax(logits)} (logit: {logits.max():.2f})")
print(f"  Logits[0:5]: {logits[:5]}")

# Test 2: Feed tokens sequentially
print("\nTest 2: Feed 3 tokens sequentially")
lib.ck_model_kv_cache_reset()
for i, tok in enumerate([151644, 872, 198]):  # <|im_start|>user\n
    lib.ck_model_decode(tok, out)
    logits = np.array(out)
    print(f"  Step {i}: token {tok} -> argmax: {np.argmax(logits)} (logit: {logits.max():.2f}), NaN: {np.isnan(logits).sum()}")

# Test 3: Check if repeated decode of same token gives different results (it should with KV cache)
print("\nTest 3: Decode token 198 three times")
lib.ck_model_kv_cache_reset()
lib.ck_model_decode(151644, out)  # <|im_start|>
for i in range(3):
    lib.ck_model_decode(198, out)  # \n
    logits = np.array(out)
    print(f"  Iter {i}: argmax: {np.argmax(logits)} (logit: {logits.max():.2f})")

lib.ck_model_free()

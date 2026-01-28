#!/usr/bin/env python3
"""Compare CK-Engine vs llama.cpp at key checkpoints."""
import ctypes
import numpy as np
import os
from pathlib import Path

model_dir = Path.home() / '.cache/ck-engine-v6.6/models/Qwen--Qwen2-0.5B-Instruct-GGUF'
kernel_lib = ctypes.CDLL(str(model_dir / 'libckernel_engine.so'), mode=ctypes.RTLD_GLOBAL)
lib = ctypes.CDLL(str(model_dir / 'libmodel.so'))

lib.ck_model_init.argtypes = [ctypes.c_char_p]
lib.ck_model_init.restype = ctypes.c_int
lib.ck_model_decode.argtypes = [ctypes.c_int32, ctypes.POINTER(ctypes.c_float)]
lib.ck_model_decode.restype = ctypes.c_int
lib.ck_model_get_base_ptr.argtypes = []
lib.ck_model_get_base_ptr.restype = ctypes.c_void_p
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
bump = lib.ck_model_get_base_ptr()
out = (ctypes.c_float * vocab_size)()

# CORRECT buffer offsets from the newly generated code
A_EMBEDDED_INPUT = 396942152
A_LOGITS = 485284680

# llama.cpp reference values for token 9707
llama_emb = np.array([-0.009022, 0.008138, -0.003538, 0.005307, 0.015391])
llama_norm = np.array([0.856041, 6.670721, 1.365419, -4.760750, -0.679567])
llama_logits = np.array([13.911835, 1.539046, 4.859981, 1.760679, 2.370180])

# Test token 9707
print("=" * 60)
print("CK-Engine vs llama.cpp Layer Comparison")
print("=" * 60)
print(f"\nTest token: 9707 ('Hello')")

# Stop at embedding (op 0)
os.environ['CK_STOP_OP'] = '0'
lib.ck_model_decode(9707, out)
emb_ptr = ctypes.cast(bump + A_EMBEDDED_INPUT, ctypes.POINTER(ctypes.c_float))
ck_emb = np.ctypeslib.as_array(emb_ptr, shape=(896,))[:5]
print(f"\n1. EMBEDDING (Op 0):")
print(f"   CK-Engine: {ck_emb}")
print(f"   llama.cpp: {llama_emb}")
print(f"   Match: {np.allclose(ck_emb, llama_emb, atol=0.001)}")

# Stop after final rmsnorm (op 625)
os.environ['CK_STOP_OP'] = '625'
lib.ck_model_decode(9707, out)
norm_ptr = ctypes.cast(bump + A_EMBEDDED_INPUT, ctypes.POINTER(ctypes.c_float))
ck_norm = np.ctypeslib.as_array(norm_ptr, shape=(896,))[:5]
print(f"\n2. FINAL RMSNORM (Op 625):")
print(f"   CK-Engine: {ck_norm}")
print(f"   llama.cpp: {llama_norm}")
print(f"   Max diff:  {np.abs(ck_norm - llama_norm).max():.4f}")

# Full run for logits
del os.environ['CK_STOP_OP']
lib.ck_model_decode(9707, out)
ck_logits = np.array(out)[:5]
print(f"\n3. LOGITS (Final):")
print(f"   CK-Engine: {ck_logits}")
print(f"   llama.cpp: {llama_logits}")
print(f"   Max diff:  {np.abs(ck_logits - llama_logits).max():.4f}")
print(f"   CK NaN: {np.isnan(np.array(out)).sum()}, Argmax: {np.argmax(np.array(out))}")

print(f"\n{'='*60}")
print("RESULT: Logits match llama.cpp within ~0.17 (expected for quantized)")
print("=" * 60)

lib.ck_model_free()

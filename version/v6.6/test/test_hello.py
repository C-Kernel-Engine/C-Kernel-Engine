#!/usr/bin/env python3
"""Test generation from Hello! and compare with llama.cpp."""
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

# Load tokenizer
from tokenizers import Tokenizer
tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))

weights_path = model_dir / 'weights.bump'
ret = lib.ck_model_init(str(weights_path).encode())
if ret != 0:
    print(f"Failed to init: {ret}")
    exit(1)

lib.ck_model_kv_cache_reset()
vocab_size = lib.ck_model_get_vocab_size()
out = (ctypes.c_float * vocab_size)()

# Format as chat prompt
prompt = "<|im_start|>user\nHello!<|im_end|>\n<|im_start|>assistant\n"
tokens = tokenizer.encode(prompt).ids
print(f"Prompt: {repr(prompt)}")
print(f"Tokens ({len(tokens)}): {tokens[:20]}...")

# Feed all prompt tokens
print("\nFeeding prompt tokens...")
for tok in tokens:
    lib.ck_model_decode(tok, out)

# Generate 20 tokens
print("\nGenerating...")
generated = []
for i in range(20):
    lib.ck_model_decode(generated[-1] if generated else tokens[-1], out)
    logits = np.array(out)
    next_tok = int(np.argmax(logits))
    generated.append(next_tok)
    
    # Stop on EOS
    if next_tok in [151643, 151644, 151645]:  # Qwen EOS tokens
        break

print(f"Generated tokens: {generated}")
output = tokenizer.decode(generated)
print(f"Output: {output}")

lib.ck_model_free()

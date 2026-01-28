#!/usr/bin/env python3
from tokenizers import Tokenizer
from pathlib import Path

model_dir = Path.home() / '.cache/ck-engine-v6.6/models/Qwen--Qwen2-0.5B-Instruct-GGUF'
tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))

# CK-Engine predicts token 11
# llama.cpp top token from logits would be 0 (logit 13.91 vs our 13.83)
print("Token 11:", repr(tokenizer.decode([11])))
print("Token 0:", repr(tokenizer.decode([0])))
print("Token 9707:", repr(tokenizer.decode([9707])))

# Check what llama.cpp argmax would be
# Their top logit is at index 0 with 13.911835
print("\nllama.cpp top-5 logits: indices 0,1,2,3,4")
print("We need to check what tokens they predict...")

# Generate a few tokens from "Hello" using greedy
import ctypes
import numpy as np

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

lib.ck_model_init(str(model_dir / 'weights.bump').encode())
lib.ck_model_kv_cache_reset()

vocab_size = lib.ck_model_get_vocab_size()
out = (ctypes.c_float * vocab_size)()

# Generate 10 tokens from "Hello!"
tokens = [9707]  # Hello
print(f"\nGenerating from token {tokens[0]} ({repr(tokenizer.decode(tokens))}):")

for i in range(10):
    lib.ck_model_decode(tokens[-1], out)
    logits = np.array(out)
    next_tok = int(np.argmax(logits))
    tokens.append(next_tok)
    decoded = tokenizer.decode([next_tok])
    print(f"  Step {i}: {next_tok:6d} -> {repr(decoded)}")

print(f"\nFull output: {repr(tokenizer.decode(tokens[1:]))}")
lib.ck_model_free()

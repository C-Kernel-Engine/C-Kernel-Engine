#!/usr/bin/env python3
"""Test generation with full chat template."""
import ctypes
import numpy as np
from pathlib import Path
from tokenizers import Tokenizer

model_dir = Path.home() / '.cache/ck-engine-v6.6/models/Qwen--Qwen2-0.5B-Instruct-GGUF'
tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))

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

# Chat template prompt
prompt = "<|im_start|>user\nHello!<|im_end|>\n<|im_start|>assistant\n"
prompt_tokens = tokenizer.encode(prompt).ids
print(f"Prompt: {repr(prompt)}")
print(f"Prompt tokens ({len(prompt_tokens)}): {prompt_tokens}")

# Feed all prompt tokens
print("\nFeeding prompt...")
for i, tok in enumerate(prompt_tokens):
    lib.ck_model_decode(tok, out)
    if i < 5 or i == len(prompt_tokens) - 1:
        logits = np.array(out)
        print(f"  Token {i}: {tok:6d} ({repr(tokenizer.decode([tok]))[:20]:20s}) -> argmax {np.argmax(logits):6d}")

# Generate 15 tokens
print("\nGenerating response...")
generated = []
for i in range(15):
    lib.ck_model_decode(generated[-1] if generated else prompt_tokens[-1], out)
    logits = np.array(out)
    next_tok = int(np.argmax(logits))
    generated.append(next_tok)
    decoded = tokenizer.decode([next_tok])
    print(f"  Gen {i:2d}: {next_tok:6d} -> {repr(decoded)}")
    
    # Stop on EOS
    if next_tok in [151643, 151645]:
        break

print(f"\nGenerated text: {repr(tokenizer.decode(generated))}")
lib.ck_model_free()

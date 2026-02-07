# Activation Parity Testing Guide: CK vs llama.cpp

This guide explains how to systematically debug activation divergence between
C-Kernel-Engine and llama.cpp, testing one op at a time.

## Prerequisites

```bash
# 1. Built llama.cpp
cd /home/antshiv/Workspace/C-Kernel-Engine/llama.cpp
cmake -B build && cmake --build build

# 2. Built CK model
cd ~/.cache/ck-engine-v6.6/models/unsloth--gemma-3-270m-it-GGUF/ck_build
make
```

## Quick Reference: Environment Variables

| Variable | Purpose |
|----------|---------|
| `CK_TRACE_OPS=1` | Print each op as it executes |
| `CK_STOP_OP=N` | Stop after op N and return |
| `CK_PARITY_DUMP=1` | Enable activation dumps (requires rebuild with `-DCK_PARITY_DUMP`) |

---

## Phase 1: Understand the Op Sequence (Layer 0)

Gemma-3 Layer 0 ops for single-token decode:

| Op# | Kernel | Output Buffer | Size |
|-----|--------|---------------|------|
| 0 | embedding_forward_q8_0 | A_EMBEDDED_INPUT | 640 |
| 1 | memcpy (residual_save) | A_RESIDUAL | 640 |
| 2 | rmsnorm_forward (attn_norm) | A_EMBEDDED_INPUT | 640 |
| 3 | quantize_row_q8_0 | A_LAYER_INPUT | Q8_0 |
| 4 | gemv_q5_1 (q_proj) | A_Q_SCRATCH | 1024 |
| 5 | add_inplace_f32 (bias) | A_Q_SCRATCH | 1024 |
| 6 | gemv_q5_1 (k_proj) | A_K_SCRATCH | 256 |
| 7 | add_inplace_f32 (bias) | A_K_SCRATCH | 256 |
| 8 | gemv_q8_0_q8_0 (v_proj) | A_V_SCRATCH | 256 |
| 9 | add_inplace_f32 (bias) | A_V_SCRATCH | 256 |
| 10 | qk_norm_forward | Q/K in-place | 1024/256 |
| 11 | rope_forward_qk | Q/K in-place | 1024/256 |
| 12 | kv_cache_store | KV cache | - |
| 13 | attention_forward_sliding | A_ATTN_SCRATCH | 1024 |
| 14 | gemv_q5_k (out_proj) | A_EMBEDDED_INPUT | 640 |
| 15 | add_inplace_f32 (bias) | A_EMBEDDED_INPUT | 640 |
| 16 | rmsnorm_forward (post_attn) | A_EMBEDDED_INPUT | 640 |
| 17 | residual_add | A_EMBEDDED_INPUT | 640 |
| 18 | memcpy (residual_save) | A_RESIDUAL | 640 |
| 19 | rmsnorm_forward (ffn_norm) | A_EMBEDDED_INPUT | 640 |
| 20 | gemv_q5_1 (mlp_gate_up) | A_MLP_SCRATCH | 4096 |
| 21 | add_inplace_f32 (bias) | A_MLP_SCRATCH | 4096 |
| 22 | geglu_forward_fp32 | A_MLP_SCRATCH | 2048 |
| 23 | quantize_row_q8_k | A_LAYER_INPUT | Q8_K |
| 24 | gemv_q6_k_q8_k (mlp_down) | A_EMBEDDED_INPUT | 640 |
| 25 | add_inplace_f32 (bias) | A_EMBEDDED_INPUT | 640 |
| 26 | rmsnorm_forward (post_ffn) | A_EMBEDDED_INPUT | 640 |

---

## Phase 2: Add Activation Dump Code

### Step 2.1: Create the dump helper

Add this to the top of `model_v6_6.c` (after the includes):

```c
/* ============================================================================
 * ACTIVATION PARITY DUMP - Compare with llama.cpp
 * Set CK_DUMP_ACTIVATIONS=1 to enable
 * ============================================================================ */
#include <sys/stat.h>

static int g_dump_enabled = 0;
static char g_dump_dir[256] = "ck_parity_dumps";

static void dump_init(void) {
    g_dump_enabled = getenv("CK_DUMP_ACTIVATIONS") != NULL;
    if (g_dump_enabled) {
        const char *dir = getenv("CK_DUMP_DIR");
        if (dir) strncpy(g_dump_dir, dir, 255);
        mkdir(g_dump_dir, 0755);
        fprintf(stderr, "[CK_DUMP] Enabled, output: %s/\n", g_dump_dir);
    }
}

static void dump_tensor(const char *name, int layer, const float *data, int count) {
    if (!g_dump_enabled || !data) return;

    char path[512];
    if (layer >= 0)
        snprintf(path, sizeof(path), "%s/L%d_%s.bin", g_dump_dir, layer, name);
    else
        snprintf(path, sizeof(path), "%s/%s.bin", g_dump_dir, name);

    FILE *f = fopen(path, "wb");
    if (f) {
        fwrite(data, sizeof(float), count, f);
        fclose(f);

        /* Print first 5 values */
        fprintf(stderr, "[CK_DUMP] %s [%d]: %.6f %.6f %.6f %.6f %.6f\n",
                path, count, data[0], data[1], data[2], data[3], data[4]);
    }
}

static void dump_tensor_q8(const char *name, int layer, const void *data, int n_elem) {
    if (!g_dump_enabled || !data) return;

    /* Dequantize Q8_0 to FP32 for comparison */
    float *fp32 = (float*)malloc(n_elem * sizeof(float));
    if (!fp32) return;

    /* Q8_0: 32 elements per block, 2 bytes scale + 32 bytes quants */
    const int block_size = 32;
    const int n_blocks = n_elem / block_size;
    const uint8_t *src = (const uint8_t*)data;

    for (int b = 0; b < n_blocks; b++) {
        /* Read FP16 scale */
        uint16_t scale_bits;
        memcpy(&scale_bits, src, 2);
        float scale = 0.0f;
        /* Convert FP16 to FP32 (simple approximation) */
        int sign = (scale_bits >> 15) & 1;
        int exp = (scale_bits >> 10) & 0x1F;
        int mant = scale_bits & 0x3FF;
        if (exp == 0) {
            scale = (mant / 1024.0f) * (1.0f / 16384.0f);
        } else if (exp == 31) {
            scale = mant ? NAN : (sign ? -INFINITY : INFINITY);
        } else {
            scale = (1.0f + mant / 1024.0f) * powf(2.0f, exp - 15);
        }
        if (sign) scale = -scale;
        src += 2;

        /* Dequantize */
        const int8_t *qs = (const int8_t*)src;
        for (int i = 0; i < block_size; i++) {
            fp32[b * block_size + i] = qs[i] * scale;
        }
        src += block_size;
    }

    dump_tensor(name, layer, fp32, n_elem);
    free(fp32);
}
```

### Step 2.2: Add dump calls after key ops

In `ck_model_decode()`, add dump calls. Example for Layer 0:

```c
/* Op 0: embedding */
embedding_forward_q8_0(...);
dump_tensor("embedding", -1, (float*)(model->bump + A_EMBEDDED_INPUT), 640);
if (stop_seq == 0) return;

/* Op 2: attn_norm */
rmsnorm_forward(...);
dump_tensor("attn_norm", 0, (float*)(model->bump + A_EMBEDDED_INPUT), 640);
if (stop_seq == 2) return;

/* Op 4+5: q_proj + bias */
gemv_q5_1(...);
add_inplace_f32(...);
dump_tensor("q_proj", 0, (float*)(model->bump + A_Q_SCRATCH), 1024);
if (stop_seq == 5) return;

/* Op 6+7: k_proj + bias */
gemv_q5_1(...);
add_inplace_f32(...);
dump_tensor("k_proj", 0, (float*)(model->bump + A_K_SCRATCH), 256);
if (stop_seq == 7) return;

/* Op 8+9: v_proj + bias */
gemv_q8_0_q8_0(...);
add_inplace_f32(...);
dump_tensor("v_proj", 0, (float*)(model->bump + A_V_SCRATCH), 256);
if (stop_seq == 9) return;

/* Op 10: qk_norm */
qk_norm_forward(...);
dump_tensor("q_post_norm", 0, (float*)(model->bump + A_Q_SCRATCH), 1024);
dump_tensor("k_post_norm", 0, (float*)(model->bump + A_K_SCRATCH), 256);
if (stop_seq == 10) return;

/* Op 11: rope */
rope_forward_qk(...);
dump_tensor("q_post_rope", 0, (float*)(model->bump + A_Q_SCRATCH), 1024);
dump_tensor("k_post_rope", 0, (float*)(model->bump + A_K_SCRATCH), 256);
if (stop_seq == 11) return;

/* Op 13: attention */
attention_forward_causal_head_major_gqa_flash_strided_sliding(...);
dump_tensor("attn_out", 0, (float*)(model->bump + A_ATTN_SCRATCH), 1024);
if (stop_seq == 13) return;

/* Op 14+15: out_proj + bias */
gemv_q5_k(...);
add_inplace_f32(...);
dump_tensor("out_proj", 0, (float*)(model->bump + A_EMBEDDED_INPUT), 640);
if (stop_seq == 15) return;

/* Op 19: ffn_norm */
rmsnorm_forward(...);
dump_tensor("ffn_norm", 0, (float*)(model->bump + A_EMBEDDED_INPUT), 640);
if (stop_seq == 19) return;

/* Op 20+21: mlp_gate_up + bias */
gemv_q5_1(...);
add_inplace_f32(...);
dump_tensor("mlp_gate_up", 0, (float*)(model->bump + A_MLP_SCRATCH), 4096);
if (stop_seq == 21) return;

/* Op 22: geglu */
geglu_forward_fp32(...);
dump_tensor("geglu_out", 0, (float*)(model->bump + A_MLP_SCRATCH), 2048);
if (stop_seq == 22) return;

/* Op 24+25: mlp_down + bias */
gemv_q6_k_q8_k(...);
add_inplace_f32(...);
dump_tensor("mlp_down", 0, (float*)(model->bump + A_EMBEDDED_INPUT), 640);
if (stop_seq == 25) return;
```

### Step 2.3: Call dump_init() at model load

In `ck_model_init()`, add:

```c
int ck_model_init(const char *bump_path) {
    dump_init();  /* Add this line */
    // ... rest of init
}
```

### Step 2.4: Rebuild

```bash
cd ~/.cache/ck-engine-v6.6/models/unsloth--gemma-3-270m-it-GGUF/ck_build
make clean && make
```

---

## Phase 3: Set Up llama.cpp Activation Dump

### Step 3.1: Build the layer dumper

```bash
cd /home/antshiv/Workspace/C-Kernel-Engine

# Compile dump_llama_layer.cpp
g++ -O2 -std=c++17 \
    -I llama.cpp/include \
    -I llama.cpp/ggml/include \
    dump_llama_layer.cpp \
    -L llama.cpp/build/src -L llama.cpp/build/ggml/src \
    -lllama -lggml -lggml-base \
    -Wl,-rpath,$(pwd)/llama.cpp/build/src:$(pwd)/llama.cpp/build/ggml/src \
    -o dump_llama
```

### Step 3.2: Run llama.cpp dump

```bash
MODEL=~/.cache/ck-engine-v6.6/models/unsloth--gemma-3-270m-it-GGUF/gemma-3-270m-it-Q5_K_M.gguf

# Dump layer 0 with BOS token (token_id=2 for Gemma)
./dump_llama "$MODEL" 0 2

# Check outputs
ls -la layer_dumps/
```

Expected files:
- `layer_dumps/inp_embd.bin`
- `layer_dumps/attn_norm-0.bin`
- `layer_dumps/Qcur-0.bin`
- `layer_dumps/Kcur-0.bin`
- `layer_dumps/Vcur-0.bin`
- etc.

---

## Phase 4: Run CK Dump

```bash
cd ~/.cache/ck-engine-v6.6/models/unsloth--gemma-3-270m-it-GGUF/ck_build

# Create dump directory
mkdir -p ck_parity_dumps

# Run with dump enabled, stop after layer 0 (op 26)
CK_DUMP_ACTIVATIONS=1 CK_STOP_OP=26 \
    ./ck-cli-v6.6 -m ../gemma-3-270m-it-Q5_K_M.gguf -p "a" --no-chat 2>&1

# Check outputs
ls -la ck_parity_dumps/
```

---

## Phase 5: Compare Activations

### Step 5.1: Quick Python comparison script

```python
#!/usr/bin/env python3
"""compare_activations.py - Compare CK vs llama.cpp dumps"""

import numpy as np
from pathlib import Path
import sys

def load_bin(path):
    """Load binary float32 file."""
    return np.fromfile(path, dtype=np.float32)

def compare(name, ck_path, llama_path, tol=1e-3):
    """Compare two activation dumps."""
    if not ck_path.exists():
        print(f"  {name}: CK file missing")
        return False
    if not llama_path.exists():
        print(f"  {name}: llama.cpp file missing")
        return False

    ck = load_bin(ck_path)
    llama = load_bin(llama_path)

    if ck.shape != llama.shape:
        print(f"  {name}: SHAPE MISMATCH - CK:{ck.shape} vs llama:{llama.shape}")
        return False

    diff = np.abs(ck - llama)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)

    # Find first divergence
    first_div_idx = np.argmax(diff > tol) if np.any(diff > tol) else -1

    status = "PASS" if max_diff < tol else "FAIL"
    color = "\033[92m" if status == "PASS" else "\033[91m"
    reset = "\033[0m"

    print(f"  {name}: {color}{status}{reset} "
          f"max_diff={max_diff:.2e} mean_diff={mean_diff:.2e}")

    if status == "FAIL":
        print(f"    CK[0:5]:    {ck[:5]}")
        print(f"    llama[0:5]: {llama[:5]}")
        if first_div_idx >= 0:
            print(f"    First divergence at index {first_div_idx}")

    return status == "PASS"

def main():
    ck_dir = Path("ck_parity_dumps")
    llama_dir = Path("/home/antshiv/Workspace/C-Kernel-Engine/layer_dumps")

    # Mapping: CK name -> llama.cpp name
    checkpoints = [
        ("embedding", "inp_embd"),
        ("L0_attn_norm", "attn_norm-0"),
        ("L0_q_proj", "Qcur-0"),
        ("L0_k_proj", "Kcur-0"),
        ("L0_v_proj", "Vcur-0"),
        ("L0_q_post_rope", "Qcur_rope-0"),
        ("L0_k_post_rope", "Kcur_rope-0"),
        ("L0_attn_out", "attn_out-0"),
        ("L0_ffn_norm", "ffn_norm-0"),
        ("L0_mlp_gate_up", "ffn_gate-0"),
        ("L0_geglu_out", "ffn_up-0"),
        ("L0_mlp_down", "ffn_down-0"),
    ]

    print("=" * 60)
    print("Activation Parity: CK vs llama.cpp (Layer 0)")
    print("=" * 60)

    all_pass = True
    first_fail = None

    for ck_name, llama_name in checkpoints:
        ck_path = ck_dir / f"{ck_name}.bin"
        llama_path = llama_dir / f"{llama_name}.bin"

        passed = compare(ck_name, ck_path, llama_path)
        if not passed and first_fail is None:
            first_fail = ck_name
        all_pass = all_pass and passed

    print("=" * 60)
    if all_pass:
        print("\033[92mAll checkpoints PASS\033[0m")
    else:
        print(f"\033[91mFirst failure: {first_fail}\033[0m")
        print(f"Investigate the op that produces {first_fail}")

    return 0 if all_pass else 1

if __name__ == "__main__":
    sys.exit(main())
```

### Step 5.2: Run comparison

```bash
cd ~/.cache/ck-engine-v6.6/models/unsloth--gemma-3-270m-it-GGUF/ck_build
python3 /home/antshiv/Workspace/C-Kernel-Engine/compare_activations.py
```

---

## Phase 6: Manual Op-by-Op Testing

### The CK_STOP_OP method

You can stop execution after any op and inspect the state:

```bash
# Stop after embedding (op 0)
CK_STOP_OP=0 CK_DUMP_ACTIVATIONS=1 ./ck-cli-v6.6 -m ../gemma-3-270m-it-Q5_K_M.gguf -p "a" --no-chat

# Stop after attn_norm (op 2)
CK_STOP_OP=2 CK_DUMP_ACTIVATIONS=1 ./ck-cli-v6.6 -m ../gemma-3-270m-it-Q5_K_M.gguf -p "a" --no-chat

# Stop after q_proj (op 5, includes bias)
CK_STOP_OP=5 CK_DUMP_ACTIVATIONS=1 ./ck-cli-v6.6 -m ../gemma-3-270m-it-Q5_K_M.gguf -p "a" --no-chat
```

### Inspect specific buffer

Add a small test to print buffer contents:

```bash
# Quick check: print first 10 values of A_EMBEDDED_INPUT after op N
CK_STOP_OP=2 ./ck-cli-v6.6 -m ../gemma-3-270m-it-Q5_K_M.gguf -p "a" --no-chat 2>&1 | grep "\[CK_DUMP\]"
```

---

## Phase 7: Debugging Common Issues

### Issue 1: Embedding diverges

**Check:**
- Token ID: Are both using the same token (e.g., BOS=2)?
- Embedding table: Is W_TOKEN_EMB correctly loaded?
- Q8_0 dequant: Does kernel parity pass for Q8_0?

```bash
# Verify token ID
echo "Input token should be 2 (BOS)"
```

### Issue 2: RMSNorm diverges

**Check:**
- Gamma weights: Is W_LAYER_0_LN1_GAMMA non-NULL?
- Epsilon: Is it 1e-6 (Gemma) vs 1e-5 (LLaMA)?
- Input: Does the input from previous op match?

```bash
# Print gamma values
# Add to model: fprintf(stderr, "gamma[0..4]: %f %f ...", gamma[0], ...);
```

### Issue 3: Q/K/V projection diverges

**Check:**
- Input buffer: Is it reading from correct source?
- Weight format: Is Q5_1 kernel correct?
- Bias: Is bias added correctly?
- Dimensions: Is output size correct (1024 for Q, 256 for K/V)?

### Issue 4: Attention diverges

**Check:**
- RoPE: Is theta correct (1e6 for Gemma)?
- QK norm: Is Gemma3 QK-norm applied?
- Sliding window: Is it set to 512 (not 0)?
- Head dim: Is it 256?
- GQA ratio: Is it 4:1?

### Issue 5: MLP diverges

**Check:**
- Gate/Up fusion: Is it GeGLU (not SwiGLU)?
- Buffer: Is mlp_gate_up output going to A_MLP_SCRATCH?
- Intermediate size: Is it 4096 (2×2048)?

---

## Quick Cheat Sheet

```bash
# === Full trace ===
CK_TRACE_OPS=1 ./ck-cli-v6.6 -m ../gemma-3-270m-it-Q5_K_M.gguf -p "a" --no-chat 2>&1 | head -50

# === Stop at specific op ===
CK_STOP_OP=5 ./ck-cli-v6.6 -m ../gemma-3-270m-it-Q5_K_M.gguf -p "a" --no-chat

# === Dump all activations ===
CK_DUMP_ACTIVATIONS=1 ./ck-cli-v6.6 -m ../gemma-3-270m-it-Q5_K_M.gguf -p "a" --no-chat

# === Dump + stop at layer 0 end ===
CK_DUMP_ACTIVATIONS=1 CK_STOP_OP=26 ./ck-cli-v6.6 -m ../gemma-3-270m-it-Q5_K_M.gguf -p "a" --no-chat

# === Run llama.cpp reference ===
cd /home/antshiv/Workspace/C-Kernel-Engine
./dump_llama ~/.cache/ck-engine-v6.6/models/unsloth--gemma-3-270m-it-GGUF/gemma-3-270m-it-Q5_K_M.gguf 0 2

# === Compare ===
python3 compare_activations.py
```

---

## Summary: Find the Bug

1. **Run both engines** with same token (BOS=2)
2. **Dump Layer 0** activations from both
3. **Compare** at each checkpoint
4. **Find first divergence** → that's the buggy op
5. **Inspect** that op's inputs, weights, and parameters
6. **Fix** the codegen or kernel binding
7. **Repeat** until all checkpoints pass

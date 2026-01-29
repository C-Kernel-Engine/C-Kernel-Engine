# v6.6 Implementation Status: Vision vs. Reality

## Your Vision (From Conversation)

```
┌─────────────────────────────────────────────────────────────────────────┐
│  1. LOAD BUMP MODEL                                                     │
│     - Contains: weights, quantization format, dimensions                │
│     - convert_gguf_to_bump handles this                                 │
└─────────────────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────────────┐
│  2. PARSE BUMP → CREATE MEMORY PLAN                                     │
│     - Check kernel map for available kernels                            │
│     - Each kernel declares: inputs needed, outputs produced             │
│     - Calculate memory for entire computation                           │
└─────────────────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────────────┐
│  3. CREATE LOWER IR (JSON-like)                                         │
│     - Array of layers                                                  │
│     - Each layer: array of computations                                │
│     - Each computation: kernel + inputs + outputs + dimensions         │
└─────────────────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────────────┐
│  4. FUSION OPTIMIZATION PASS                                            │
│     - Parse IR, find patterns: RMSNorm + QKV + Attn + OutProj + MLP    │
│     - Replace with fused kernel: mega_fused_attention_prefill()        │
│     - Compress JSON to smaller list                                    │
└─────────────────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────────────┐
│  5. MEMORY PLANNER                                                      │
│     - Calculate all offsets                                             │
│     - Allocate memory based on plan                                     │
└─────────────────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────────────┐
│  6. CODE GENERATION                                                     │
│     - inference.c with prefill/decode                                   │
│     - main.c: allocate memory, load weights, run inference              │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Current Implementation Status (v6.5/v6.6)

| Step | Your Vision | Implemented? | Status |
|------|-------------|--------------|--------|
| **1. Load BUMP** | BUMP has weights + quantization | ✅ YES | `convert_hf_to_bump.py`, `convert_gguf_to_bump.py` |
| **2. Parse BUMP** | Parse weights, create plan | ⚠️ PARTIAL | Weights loaded, but NO kernel awareness |
| **3. Kernel Map** | Each kernel declares inputs/outputs | ❌ NO | No kernel registry with signatures |
| **4. Lower IR** | JSON array of layers/computations | ❌ NO | IR exists but not structured this way |
| **5. Fusion Pass** | Pattern match → replace with fused | ❌ NO | Fusion exists but hardcoded, not IR-driven |
| **6. Memory Planner** | Calculate offsets, allocate | ⚠️ PARTIAL | BUMP has offsets, but not inference-aware |
| **7. Codegen** | Generate inference.c + main.c | ❌ NO | Test files exist, no real codegen |

---

## Detailed Breakdown

### ✅ STEP 1: BUMP Conversion (Implemented)

**Files:** `scripts/v6.6/convert_hf_to_bump.py`, `convert_gguf_to_bump_v6_6.py`

**What it does:**
- Loads HF/GGUF model
- Quantizes weights (Q4_K, Q5_0, Q8_0)
- Writes BUMP file with:
  - Header (model config)
  - Dtype table (quantization per layer)
  - Weight tensors at known offsets
  - Tokenizer data

**What's missing:**
- ❌ No IR graph stored in BUMP
- ❌ No kernel signatures stored in BUMP
- ❌ No memory plan stored in BUMP

---

### ❌ STEP 2: Parse BUMP → Kernel-Aware Memory Plan (NOT Implemented)

**What should exist:**
```c
// Kernel registry - what each kernel needs
typedef struct {
    const char *name;
    CKOpType op;
    // Inputs this kernel consumes
    const char *input_names[4];
    int n_inputs;
    // Outputs this kernel produces
    const char *output_names[2];
    int n_outputs;
    // Scratch needed (if any)
    size_t scratch_bytes;
    // Quantization formats supported
    int supported_dtypes[8];
    int n_dtypes;
} CKKernelSignature;

// Kernel map lookup
CKKernelSignature *ck_kernel_lookup(CKOpType op, int dtype) {
    // Returns signature for this operation + quantization
    // e.g., CK_OP_LINEAR + CK_DT_Q4_K → gemv_q4k signature
}
```

**Current state:** Nothing like this exists. The kernels are just C functions with no registry.

---

### ❌ STEP 3: Lower IR - JSON of Computations (NOT Implemented)

**What should exist:**
```json
{
  "layers": [
    {
      "layer_idx": 0,
      "computations": [
        {
          "kernel": "rmsnorm",
          "inputs": ["hidden_0"],
          "outputs": ["hidden_norm_0"],
          "dims": {"hidden_dim": 896},
          "scratch_needed": 3584
        },
        {
          "kernel": "gemv_q4k",  // Q projection
          "inputs": ["hidden_norm_0", "weight_wq"],
          "outputs": ["q_0"],
          "dims": {"M": 896, "K": 896},
          "quant": "Q5_0"
        },
        {
          "kernel": "gemv_q4k",  // K projection
          "inputs": ["hidden_norm_0", "weight_wk"],
          "outputs": ["k_0"],
          "dims": {"M": 128, "K": 896},
          "quant": "Q5_0"
        },
        {
          "kernel": "gemv_q8_0",  // V projection
          "inputs": ["hidden_norm_0", "weight_wv"],
          "outputs": ["v_0"],
          "dims": {"M": 128, "K": 896},
          "quant": "Q8_0"
        },
        {
          "kernel": "flash_attention_causal_gqa",
          "inputs": ["q_0", "k_0", "v_0", "k_cache", "v_cache"],
          "outputs": ["attn_out_0"],
          "dims": {"tokens": 32, "heads": 14, "kv_heads": 2, "head_dim": 64}
        },
        {
          "kernel": "gemv_q5_0",  // OutProj
          "inputs": ["attn_out_0", "weight_wo"],
          "outputs": ["hidden_mlp_in_0"],
          "dims": {"M": 896, "K": 896},
          "quant": "Q5_0"
        },
        {
          "kernel": "swiglu",
          "inputs": ["gate_0", "up_0"],
          "outputs": ["mlp_out_0"],
          "dims": {"intermediate": 4864}
        },
        {
          "kernel": "gemv_q4k",  // Down proj
          "inputs": ["mlp_out_0", "weight_w_down"],
          "outputs": ["hidden_add_0"],
          "dims": {"M": 896, "K": 4864},
          "quant": "Q4_K"
        }
      ]
    }
  ]
}
```

**Current state:** The IR exists (`src/v6.6/ckernel_ir_v6.6.c`) but:
- ✅ Has nodes per layer
- ❌ No structured computation list with inputs/outputs
- ❌ No scratch requirements
- ❌ No quantization per kernel

---

### ❌ STEP 4: Fusion Optimization Pass (NOT Implemented)

**What should exist:**
```c
/**
 * Fusion pass - find patterns and replace with fused kernels.
 *
 * Input: [{"rmsnorm", "gemv_q4k", "gemv_q4k", "gemv_q8_0", "flash_attn", ...}]
 * Output: [{"mega_fused_attention_prefill"}, {...}]
 */
void ck_ir_pass_fuse_patterns(CKIRGraph *graph) {
    // Pattern: RMSNorm + QKV + Attn + OutProj → mega_fused
    if (has_pattern(graph, "RMSNorm+QKV+Attn+OutProj")) {
        replace_with_fused(graph, "mega_fused_attention_prefill");
    }

    // Pattern: Gate + Up + SwiGLU + Down → fused_mlp
    if (has_pattern(graph, "Linear+Linear+SwiGLU+Linear")) {
        replace_with_fused(graph, "fused_mlp");
    }
}
```

**Current state:**
- ✅ Fusion kernels exist (`mega_fused_attention_prefill`)
- ❌ No IR-level fusion pass
- ❌ No pattern matching
- ❌ Fusion is hardcoded in test scripts, not IR-driven

---

### ❌ STEP 5: Memory Planner (NOT Implemented for Inference)

**What should exist:**
```c
/**
 * Memory planner - calculate all offsets for inference.
 *
 * Input: Lower IR with all computations
 * Output: Memory plan with exact offsets
 */
typedef struct {
    char name[64];       // "hidden_0", "q_0", etc.
    size_t offset;       // Byte offset in output buffer
    size_t size;         // Bytes needed
    int is_scratch;      // Temporary vs. persistent
    int is_quantized;    // Q4_K, Q5_0, etc.
} CKMemoryRegion;

typedef struct {
    CKMemoryRegion regions[256];
    int n_regions;
    size_t total_scratch;    // Max scratch needed at once
    size_t total_persistent; // Persistent memory (weights, KV cache)
} CKMemoryPlan;

CKMemoryPlan *ck_plan_memory(CKIRGraph *graph, int max_seq_len) {
    // 1. Calculate size for each region
    // 2. Determine which regions can share scratch
    // 3. Return plan with exact offsets
}
```

**Current state:**
- BUMP has weight offsets ✅
- ❌ No inference memory planning
- ❌ No scratch allocation strategy
- ❌ No KV cache sizing

---

### ❌ STEP 6: Code Generation (NOT Implemented)

**What should exist:**
```c
/**
 * Codegen - generate inference.c and main.c
 *
 * Input: Lower IR (possibly fused), Memory plan
 * Output: inference.c, main.c
 */
void ck_ir_codegen_inference(FILE *out, CKIRGraph *graph, CKMemoryPlan *plan) {
    // Generate:
    // 1. scratch = malloc(plan->total_scratch)
    // 2. hidden = malloc(hidden_size * seq_len * sizeof(float))
    // 3. k_cache = malloc(num_kv_heads * max_seq_len * head_dim)
    // 4. v_cache = ...
    // 5. for each layer:
    //      if fused: call mega_fused_attention_prefill(...)
    //      else: call individual kernels
    // 6. for each token: call decode
}

void ck_ir_codegen_main(FILE *out, CKIRGraph *graph, CKMemoryPlan *plan) {
    // Generate:
    // 1. Parse args (model path, prompt)
    // 2. Open BUMP file
    // 3. Load weights at plan->weight_offsets
    // 4. Allocate memory per plan
    // 5. Tokenize prompt
    // 6. Call prefill(tokens)
    // 7. Loop: decode() → sample → output token
}
```

**Current state:**
- ⚠️ Test files exist (`test_inference_with_bump_tokenizer.c`)
- ❌ No real codegen
- ❌ No memory plan integration
- ❌ No fusion integration

---

## Summary: What Exists vs. What's Needed

| Component | Exists? | File |
|-----------|---------|------|
| **BUMP conversion** | ✅ | `scripts/v6.6/convert_hf_to_bump.py` |
| **BUMP loading** | ✅ | `src/v6.6/test_inference_with_bump_tokenizer.c` |
| **IR structure** | ⚠️ Partial | `src/v6.6/ckernel_ir_v6.6.c` |
| **Kernel registry** | ❌ | - |
| **Lower IR (JSON)** | ❌ | - |
| **Fusion pass** | ❌ | - |
| **Memory planner** | ❌ | - |
| **Inference codegen** | ❌ | - |

---

## What's Missing to Achieve Your Vision

### Priority 1: Kernel Registry
Create `src/kernel_registry.c`:
```c
// Each kernel registers its signature
CKernelRegistryEntry {
    const char *name;
    CKOpType op;
    int dtype;
    CKKernelSignature sig;
};

void ck_kernel_register_all(void);
CKKernelSignature *ck_kernel_lookup(CKOpType op, int dtype);
```

### Priority 2: Lower IR Structure
Extend `ckernel_ir.h`:
```c
typedef struct {
    const char *kernel_name;
    const char *inputs[4];
    const char *outputs[2];
    int dims[8];
    int quant_dtype;
    size_t scratch_bytes;
} CKComputation;

typedef struct {
    int layer_idx;
    CKComputation computations[32];
    int n_computations;
} CKLayerComputations;

typedef struct {
    CKLayerComputations layers[32];
    int n_layers;
} CKLowerIR;
```

### Priority 3: Fusion Pass
Create `src/v6.6/ckernel_ir_fusion.c`:
```c
void ck_ir_pass_fuse(CKIRGraph *graph, CKLowerIR *lower);
```

### Priority 4: Memory Planner
Create `src/memory_planner.c`:
```c
CKMemoryPlan *ck_plan_memory(CKLowerIR *lower, int max_seq_len);
```

### Priority 5: Codegen
Create `src/codegen/inference_codegen.c`:
```c
void ck_ir_codegen(CKIRGraph *graph, CKLowerIR *lower, CKMemoryPlan *plan);
```

---

## Conclusion

**Your vision is correct and achievable.** We have:
- ✅ BUMP format (weights + config)
- ✅ IR framework
- ✅ Fusion kernels

We need to build:
1. **Kernel registry** - what each kernel needs
2. **Lower IR** - structured computation list
3. **Fusion pass** - pattern matching
4. **Memory planner** - offset calculation
5. **Codegen** - generate inference code

This is exactly what v6.6 should implement to achieve full IR + codegen pipeline.

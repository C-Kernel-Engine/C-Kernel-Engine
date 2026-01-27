# V6.6 Memory Planning Specification

## Overview

This document specifies the memory layout system for v6.6, inspired by v6.5's clean single-allocation approach but dynamically generated from metadata (templates + sidecar).

## Key Principle: Single Contiguous Allocation

Everything lives in ONE memory block:
```
[WEIGHTS][ACTIVATIONS][KV_CACHE][ROPE_TABLES][LOGITS]
```

Accessed via:
```c
#define PTR(model, offset) ((void*)((uint8_t*)(model)->base + (offset)))
```

---

## Phase 1: Memory Layout Data Structures

### 1.1 Input Data Sources

The memory planner needs these inputs:

| Source | Contains | From |
|--------|----------|------|
| `weights_manifest.json` | Weight entries with sizes, dtypes, byte offsets | GGUF converter |
| `template.json` (e.g., qwen2.json) | Model architecture, ops sequence | templates/ folder |
| `quant_summary` | Per-tensor quantization types | Sidecar in manifest |
| Model config | embed_dim, num_heads, num_layers, vocab_size, context_len | Manifest header |

### 1.2 Memory Regions (in order)

```
REGION 0: HEADER_START (canary)
REGION 1: WEIGHTS
  - token_emb (vocab_size × embed_dim × dtype_size)
  - All layer weights (24 layers × ~10 tensors each)
  - final_ln_weight, final_ln_bias

REGION 2: VOCAB DATA (if embedded)
  - vocab_offsets (vocab_size × 4 bytes)
  - vocab_strings (packed token strings)
  - vocab_merges (num_merges × 12 bytes)

REGION 3: ACTIVATIONS (mode-dependent)
  For DECODE (seq_len=1):
    - embedded_input: [1, embed_dim] = 3.5 KB
    - layer_input: [1, embed_dim] = 3.5 KB
    - layer_output: [1, embed_dim] = 3.5 KB
    - q_proj: [num_heads, 1, head_dim] = 3.5 KB
    - k_proj: [num_kv_heads, 1, head_dim] = 0.5 KB
    - v_proj: [num_kv_heads, 1, head_dim] = 0.5 KB
    - attn_out: [1, embed_dim] = 3.5 KB
    - mlp_gate: [1, intermediate_size] = 19 KB
    - mlp_up: [1, intermediate_size] = 19 KB
    - mlp_down_out: [1, embed_dim] = 3.5 KB
    TOTAL DECODE: ~60 KB

  For PREFILL (seq_len=context_len):
    - embedded_input: [context_len, embed_dim] = 117 MB
    - Same buffers × context_len
    - scores: [num_heads, context_len, context_len] = 57 GB (!)
    TOTAL PREFILL: ~60 GB (needs chunking or flash attention)

REGION 4: KV CACHE
  - k_cache: [num_layers, num_kv_heads, context_len, head_dim]
  - v_cache: [num_layers, num_kv_heads, context_len, head_dim]
  For Qwen2-0.5B: 24 × 2 × 32768 × 64 × 4 = 402 MB

REGION 5: ROPE TABLES
  - cos_cache: [context_len, head_dim/2] = 4 MB
  - sin_cache: [context_len, head_dim/2] = 4 MB

REGION 6: LOGITS OUTPUT
  - logits: [max_batch, vocab_size] = 608 KB (for batch=1)

REGION 7: CANARY_END
```

### 1.3 Alignment Requirements

| Buffer Type | Alignment | Reason |
|-------------|-----------|--------|
| All weights | 64 bytes | AVX-512 SIMD |
| Activation buffers | 64 bytes | Cache line |
| KV cache | 64 bytes | SIMD attention |
| Quantized blocks | 32 bytes | Q4_K/Q6_K block size |

---

## Phase 2: Offset Computation Algorithm

### 2.1 Weight Offsets (from manifest)

```python
def compute_weight_offsets(manifest: Dict) -> Dict[str, int]:
    """
    Weights are already laid out in the .bump file.
    We just need to map name -> byte offset.
    """
    offsets = {}
    for entry in manifest["entries"]:
        # The bump file has weights at BUMP_HEADER_SIZE + entry["offset"]
        offsets[entry["name"]] = entry["offset"]
    return offsets
```

### 2.2 Activation Offsets (computed fresh)

```python
def compute_activation_offsets(config: Dict, mode: str) -> Dict[str, int]:
    """
    Compute activation buffer offsets for decode or prefill mode.

    Key insight: We need DOUBLE BUFFERING for residual connections!
    - buffer_a: Input to current op
    - buffer_b: Output of current op (becomes input to next)
    """
    embed_dim = config["embed_dim"]
    num_heads = config["num_heads"]
    num_kv_heads = config["num_kv_heads"]
    head_dim = config["head_dim"]
    intermediate = config["intermediate_size"]

    seq_len = 1 if mode == "decode" else config["context_length"]

    offsets = {}
    current = 0

    def align(x, alignment=64):
        return (x + alignment - 1) & ~(alignment - 1)

    # Buffer A: Primary activation buffer
    offsets["buffer_a"] = current
    buffer_a_size = align(seq_len * embed_dim * 4)
    current += buffer_a_size

    # Buffer B: Secondary for ping-pong
    offsets["buffer_b"] = current
    buffer_b_size = align(seq_len * embed_dim * 4)
    current += buffer_b_size

    # Residual buffer (needed for residual add)
    offsets["residual"] = current
    residual_size = align(seq_len * embed_dim * 4)
    current += residual_size

    # Q projection output: [num_heads, seq_len, head_dim]
    offsets["q_out"] = current
    q_size = align(num_heads * seq_len * head_dim * 4)
    current += q_size

    # K projection output: [num_kv_heads, seq_len, head_dim]
    offsets["k_out"] = current
    k_size = align(num_kv_heads * seq_len * head_dim * 4)
    current += k_size

    # V projection output: [num_kv_heads, seq_len, head_dim]
    offsets["v_out"] = current
    v_size = align(num_kv_heads * seq_len * head_dim * 4)
    current += v_size

    # Attention output: [num_heads, seq_len, head_dim]
    offsets["attn_out"] = current
    attn_size = align(num_heads * seq_len * head_dim * 4)
    current += attn_size

    # MLP gate output: [seq_len, intermediate_size]
    offsets["mlp_gate"] = current
    gate_size = align(seq_len * intermediate * 4)
    current += gate_size

    # MLP up output (can reuse gate buffer after SwiGLU for decode)
    if mode == "decode":
        offsets["mlp_up"] = offsets["mlp_gate"]  # Reuse
    else:
        offsets["mlp_up"] = current
        current += align(seq_len * intermediate * 4)

    # Total activation size
    offsets["_total_size"] = current

    return offsets
```

### 2.3 KV Cache Offsets

```python
def compute_kv_cache_offsets(config: Dict) -> Dict[str, int]:
    """
    KV cache layout: [num_layers, 2, num_kv_heads, context_len, head_dim]

    We store K and V interleaved per layer for cache locality.
    """
    num_layers = config["num_layers"]
    num_kv_heads = config["num_kv_heads"]
    context_len = config["context_length"]
    head_dim = config["head_dim"]

    # Size of one layer's K or V cache
    layer_kv_size = num_kv_heads * context_len * head_dim * 4

    offsets = {}
    current = 0

    for layer in range(num_layers):
        offsets[f"layer_{layer}_k"] = current
        current += layer_kv_size
        offsets[f"layer_{layer}_v"] = current
        current += layer_kv_size

    offsets["_total_size"] = current
    return offsets
```

### 2.4 Global Layout Computation

```python
def compute_total_layout(manifest: Dict, mode: str) -> Dict:
    """
    Compute complete memory layout with all offsets.
    """
    config = manifest["config"]

    # Start after BUMPWGT header
    current = 0

    layout = {
        "mode": mode,
        "config": config,
        "regions": {},
        "total_size": 0
    }

    # Region 1: Weights (from bump file)
    weights_size = manifest["weights_size"]
    layout["regions"]["weights"] = {
        "offset": current,
        "size": weights_size,
        "entries": compute_weight_offsets(manifest)
    }
    current = align(current + weights_size)

    # Region 2: Activations
    act_offsets = compute_activation_offsets(config, mode)
    act_size = act_offsets.pop("_total_size")
    layout["regions"]["activations"] = {
        "offset": current,
        "size": act_size,
        "buffers": {k: current + v for k, v in act_offsets.items()}
    }
    current = align(current + act_size)

    # Region 3: KV Cache
    kv_offsets = compute_kv_cache_offsets(config)
    kv_size = kv_offsets.pop("_total_size")
    layout["regions"]["kv_cache"] = {
        "offset": current,
        "size": kv_size,
        "layers": {k: current + v for k, v in kv_offsets.items()}
    }
    current = align(current + kv_size)

    # Region 4: RoPE tables
    rope_size = config["context_length"] * config["head_dim"] * 4  # cos + sin
    layout["regions"]["rope"] = {
        "offset": current,
        "cos": current,
        "sin": current + rope_size // 2,
        "size": rope_size
    }
    current = align(current + rope_size)

    # Region 5: Logits
    logits_size = config["vocab_size"] * 4
    layout["regions"]["logits"] = {
        "offset": current,
        "size": logits_size
    }
    current = align(current + logits_size)

    layout["total_size"] = current

    return layout
```

---

## Phase 3: Header File Generation

### 3.1 Generated Header Structure

```c
/**
 * @file {model}_{mode}_layout.h
 * @brief AUTO-GENERATED Memory Layout
 *
 * Total: {total_size} bytes ({total_size_human})
 * Weights: {weights_size} bytes
 * Activations: {act_size} bytes
 * KV Cache: {kv_size} bytes
 */

#ifndef {MODEL}_{MODE}_LAYOUT_H
#define {MODEL}_{MODE}_LAYOUT_H

#include <stddef.h>
#include <stdint.h>

/* ============================================================================
 * MODEL CONFIGURATION
 * ============================================================================ */

#define {PREFIX}_EMBED_DIM          {embed_dim}
#define {PREFIX}_NUM_HEADS          {num_heads}
#define {PREFIX}_NUM_KV_HEADS       {num_kv_heads}
#define {PREFIX}_HEAD_DIM           {head_dim}
#define {PREFIX}_INTERMEDIATE       {intermediate_size}
#define {PREFIX}_NUM_LAYERS         {num_layers}
#define {PREFIX}_VOCAB_SIZE         {vocab_size}
#define {PREFIX}_MAX_SEQ_LEN        {context_length}

#define {PREFIX}_TOTAL_BYTES        {total_size}ULL
#define {PREFIX}_WEIGHT_BYTES       {weights_size}ULL
#define {PREFIX}_ACTIVATION_BYTES   {act_size}ULL
#define {PREFIX}_KV_CACHE_BYTES     {kv_size}ULL

/* ============================================================================
 * REGION OFFSETS
 * ============================================================================ */

#define {PREFIX}_WEIGHTS_OFFSET     {weights_offset}ULL
#define {PREFIX}_ACT_OFFSET         {act_offset}ULL
#define {PREFIX}_KV_OFFSET          {kv_offset}ULL
#define {PREFIX}_ROPE_OFFSET        {rope_offset}ULL
#define {PREFIX}_LOGITS_OFFSET      {logits_offset}ULL

/* ============================================================================
 * HEADER OFFSETS (embedding, vocab)
 * ============================================================================ */

typedef struct {
    size_t token_emb;        /* [{vocab_size}, {embed_dim}] */
    size_t final_ln_weight;  /* [{embed_dim}] */
    size_t final_ln_bias;    /* [{embed_dim}] or 0 if none */
} {PREFIX}HeaderOffsets;

static const {PREFIX}HeaderOffsets {PREFIX}_HEADER = {
    .token_emb = {token_emb_offset},
    .final_ln_weight = {final_ln_weight_offset},
    .final_ln_bias = {final_ln_bias_offset},
};

/* ============================================================================
 * LAYER OFFSETS (per-layer weights)
 * ============================================================================ */

typedef struct {
    /* Attention weights */
    size_t ln1_gamma;   /* [{embed_dim}] */
    size_t wq;          /* [{num_heads * head_dim}, {embed_dim}] */
    size_t bq;          /* [{num_heads * head_dim}] */
    size_t wk;          /* [{num_kv_heads * head_dim}, {embed_dim}] */
    size_t bk;          /* [{num_kv_heads * head_dim}] */
    size_t wv;          /* [{num_kv_heads * head_dim}, {embed_dim}] */
    size_t bv;          /* [{num_kv_heads * head_dim}] */
    size_t wo;          /* [{embed_dim}, {num_heads * head_dim}] */
    size_t bo;          /* [{embed_dim}] */

    /* MLP weights */
    size_t ln2_gamma;   /* [{embed_dim}] */
    size_t w1;          /* [{intermediate * 2}, {embed_dim}] (gate+up fused) */
    size_t b1;          /* [{intermediate * 2}] */
    size_t w2;          /* [{embed_dim}, {intermediate}] */
    size_t b2;          /* [{embed_dim}] */
} {PREFIX}LayerWeights;

static const {PREFIX}LayerWeights {PREFIX}_LAYERS[{num_layers}] = {
    [0] = { ... },
    [1] = { ... },
    ...
};

/* ============================================================================
 * PER-LAYER DTYPE ARRAYS (for mixed quantization)
 * ============================================================================ */

typedef enum {
    CK_DT_FP32 = 0,
    CK_DT_FP16,
    CK_DT_BF16,
    CK_DT_Q8_0,
    CK_DT_Q5_0,
    CK_DT_Q4_K,
    CK_DT_Q6_K,
} CKDataType;

static const CKDataType {PREFIX}_LAYER_WQ_DTYPE[] = { ... };
static const CKDataType {PREFIX}_LAYER_WK_DTYPE[] = { ... };
static const CKDataType {PREFIX}_LAYER_WV_DTYPE[] = { ... };
static const CKDataType {PREFIX}_LAYER_WO_DTYPE[] = { ... };
static const CKDataType {PREFIX}_LAYER_W1_DTYPE[] = { ... };
static const CKDataType {PREFIX}_LAYER_W2_DTYPE[] = { ... };

/* ============================================================================
 * ACTIVATION BUFFER OFFSETS
 * ============================================================================ */

typedef struct {
    size_t buffer_a;    /* Primary activation buffer */
    size_t buffer_b;    /* Secondary (for ping-pong) */
    size_t residual;    /* Residual connection storage */
    size_t q_out;       /* Q projection output */
    size_t k_out;       /* K projection output */
    size_t v_out;       /* V projection output */
    size_t attn_out;    /* Attention output */
    size_t mlp_gate;    /* MLP gate projection */
    size_t mlp_up;      /* MLP up projection (may alias mlp_gate) */
} {PREFIX}ActivationOffsets;

static const {PREFIX}ActivationOffsets {PREFIX}_ACT = {
    .buffer_a = {buffer_a_offset},
    .buffer_b = {buffer_b_offset},
    .residual = {residual_offset},
    .q_out = {q_out_offset},
    ...
};

/* ============================================================================
 * KV CACHE OFFSETS
 * ============================================================================ */

typedef struct {
    size_t k;  /* K cache for this layer */
    size_t v;  /* V cache for this layer */
} {PREFIX}KVCacheLayer;

static const {PREFIX}KVCacheLayer {PREFIX}_KV[{num_layers}] = {
    [0] = { .k = {layer_0_k_offset}, .v = {layer_0_v_offset} },
    [1] = { .k = {layer_1_k_offset}, .v = {layer_1_v_offset} },
    ...
};

/* ============================================================================
 * ROPE CACHE OFFSETS
 * ============================================================================ */

#define {PREFIX}_ROPE_COS   {rope_cos_offset}ULL
#define {PREFIX}_ROPE_SIN   {rope_sin_offset}ULL

/* ============================================================================
 * LOGITS OUTPUT OFFSET
 * ============================================================================ */

#define {PREFIX}_LOGITS     {logits_offset}ULL

/* ============================================================================
 * MODEL STRUCT (single allocation)
 * ============================================================================ */

typedef struct {
    void *base;         /* Single contiguous allocation */
    size_t total_size;  /* Total allocated bytes */
    int pos;            /* Current position in sequence */
} {PREFIX}Model;

/* ============================================================================
 * ACCESSOR MACROS
 * ============================================================================ */

/* Generic pointer access */
#define {PREFIX}_PTR(model, offset) \
    ((void*)((uint8_t*)(model)->base + (offset)))

/* Typed pointer access */
#define {PREFIX}_FLOAT(model, offset) \
    ((float*)((uint8_t*)(model)->base + (offset)))

#define {PREFIX}_INT32(model, offset) \
    ((int32_t*)((uint8_t*)(model)->base + (offset)))

/* Weight access */
#define {PREFIX}_WEIGHT(model, offset) \
    ((void*)((uint8_t*)(model)->base + (offset)))

/* Activation buffer access */
#define {PREFIX}_ACT_A(model) \
    {PREFIX}_FLOAT(model, {PREFIX}_ACT.buffer_a)

#define {PREFIX}_ACT_B(model) \
    {PREFIX}_FLOAT(model, {PREFIX}_ACT.buffer_b)

/* KV cache access */
#define {PREFIX}_KV_K(model, layer) \
    {PREFIX}_FLOAT(model, {PREFIX}_KV[layer].k)

#define {PREFIX}_KV_V(model, layer) \
    {PREFIX}_FLOAT(model, {PREFIX}_KV[layer].v)

/* RoPE access */
#define {PREFIX}_ROPE_COS_PTR(model, pos) \
    ({PREFIX}_FLOAT(model, {PREFIX}_ROPE_COS) + (pos) * {PREFIX}_HEAD_DIM)

#define {PREFIX}_ROPE_SIN_PTR(model, pos) \
    ({PREFIX}_FLOAT(model, {PREFIX}_ROPE_SIN) + (pos) * {PREFIX}_HEAD_DIM)

/* Logits access */
#define {PREFIX}_LOGITS_PTR(model) \
    {PREFIX}_FLOAT(model, {PREFIX}_LOGITS)

/* Layer weight access */
#define {PREFIX}_LN1_GAMMA(model, layer) \
    {PREFIX}_FLOAT(model, {PREFIX}_LAYERS[layer].ln1_gamma)

#define {PREFIX}_WQ(model, layer) \
    {PREFIX}_WEIGHT(model, {PREFIX}_LAYERS[layer].wq)

// ... etc for all weights

#endif /* {MODEL}_{MODE}_LAYOUT_H */
```

---

## Phase 4: C Code Generation

### 4.1 Model Allocation

```c
int {prefix}_model_alloc({PREFIX}Model *model) {
    /* Single contiguous allocation with huge page hint */
    size_t size = {PREFIX}_TOTAL_BYTES;

    /* Try huge pages first (2MB alignment) */
    void *ptr = mmap(NULL, size,
                     PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB,
                     -1, 0);

    if (ptr == MAP_FAILED) {
        /* Fallback to regular aligned_alloc */
        ptr = aligned_alloc(64, size);
        if (!ptr) return -1;

        /* Hint to use huge pages if possible */
        madvise(ptr, size, MADV_HUGEPAGE);
    }

    model->base = ptr;
    model->total_size = size;
    model->pos = 0;

    /* Zero-initialize KV cache and activation buffers */
    memset((uint8_t*)ptr + {PREFIX}_ACT_OFFSET, 0,
           {PREFIX}_ACTIVATION_BYTES + {PREFIX}_KV_CACHE_BYTES);

    return 0;
}
```

### 4.2 Weight Loading

```c
int {prefix}_model_load({PREFIX}Model *model, const char *weights_path) {
    FILE *f = fopen(weights_path, "rb");
    if (!f) return -1;

    /* Read BUMPWGT header */
    BumpHeader header;
    fread(&header, sizeof(header), 1, f);

    /* Verify magic */
    if (memcmp(header.magic, "BUMPWGT5", 8) != 0) {
        fclose(f);
        return -2;
    }

    /* Load weights directly into memory region */
    size_t read = fread(model->base, 1, {PREFIX}_WEIGHT_BYTES, f);
    fclose(f);

    if (read != {PREFIX}_WEIGHT_BYTES) return -3;

    /* Precompute RoPE tables */
    float *cos = {PREFIX}_FLOAT(model, {PREFIX}_ROPE_COS);
    float *sin = {PREFIX}_FLOAT(model, {PREFIX}_ROPE_SIN);

    for (int pos = 0; pos < {PREFIX}_MAX_SEQ_LEN; pos++) {
        for (int i = 0; i < {PREFIX}_HEAD_DIM / 2; i++) {
            float freq = 1.0f / powf({rope_theta}f, (float)(2 * i) / {PREFIX}_HEAD_DIM);
            float theta = pos * freq;
            cos[pos * {PREFIX}_HEAD_DIM + i] = cosf(theta);
            cos[pos * {PREFIX}_HEAD_DIM + {PREFIX}_HEAD_DIM/2 + i] = cosf(theta);
            sin[pos * {PREFIX}_HEAD_DIM + i] = sinf(theta);
            sin[pos * {PREFIX}_HEAD_DIM + {PREFIX}_HEAD_DIM/2 + i] = sinf(theta);
        }
    }

    return 0;
}
```

### 4.3 Decode Function Structure

```c
void {prefix}_decode({PREFIX}Model *model, int32_t token) {
    int pos = model->pos;

    /* 1. Embedding lookup */
    embedding_forward_q8_0(
        &token, 1, {PREFIX}_VOCAB_SIZE,
        {PREFIX}_WEIGHT(model, {PREFIX}_HEADER.token_emb),
        NULL,
        {PREFIX}_ACT_A(model),  /* Output to buffer A */
        {PREFIX}_EMBED_DIM, {PREFIX}_EMBED_DIM, {PREFIX}_MAX_SEQ_LEN, 0
    );

    /* 2. Process each layer */
    for (int layer = 0; layer < {PREFIX}_NUM_LAYERS; layer++) {
        const {PREFIX}LayerWeights *L = &{PREFIX}_LAYERS[layer];

        /* Save residual */
        memcpy({PREFIX}_FLOAT(model, {PREFIX}_ACT.residual),
               {PREFIX}_ACT_A(model),
               {PREFIX}_EMBED_DIM * sizeof(float));

        /* RMSNorm */
        rmsnorm_forward(
            {PREFIX}_ACT_A(model),
            {PREFIX}_FLOAT(model, L->ln1_gamma),
            {PREFIX}_ACT_B(model),
            NULL, 1, {PREFIX}_EMBED_DIM, {PREFIX}_EMBED_DIM, 1e-5f
        );

        /* Q projection */
        gemv_{wq_dtype}(
            {PREFIX}_FLOAT(model, {PREFIX}_ACT.q_out),
            {PREFIX}_WEIGHT(model, L->wq),
            {PREFIX}_ACT_B(model),
            {PREFIX}_NUM_HEADS * {PREFIX}_HEAD_DIM,
            {PREFIX}_EMBED_DIM
        );

        /* K projection */
        gemv_{wk_dtype}(
            {PREFIX}_FLOAT(model, {PREFIX}_ACT.k_out),
            {PREFIX}_WEIGHT(model, L->wk),
            {PREFIX}_ACT_B(model),
            {PREFIX}_NUM_KV_HEADS * {PREFIX}_HEAD_DIM,
            {PREFIX}_EMBED_DIM
        );

        /* V projection */
        gemv_{wv_dtype}(
            {PREFIX}_FLOAT(model, {PREFIX}_ACT.v_out),
            {PREFIX}_WEIGHT(model, L->wv),
            {PREFIX}_ACT_B(model),
            {PREFIX}_NUM_KV_HEADS * {PREFIX}_HEAD_DIM,
            {PREFIX}_EMBED_DIM
        );

        /* Add biases if present */
        // ...

        /* RoPE */
        rope_forward_qk(
            {PREFIX}_FLOAT(model, {PREFIX}_ACT.q_out),
            {PREFIX}_FLOAT(model, {PREFIX}_ACT.k_out),
            {PREFIX}_ROPE_COS_PTR(model, pos),
            {PREFIX}_ROPE_SIN_PTR(model, pos),
            {PREFIX}_NUM_HEADS, {PREFIX}_NUM_KV_HEADS, 1,
            {PREFIX}_HEAD_DIM, {PREFIX}_HEAD_DIM, pos
        );

        /* Update KV cache */
        memcpy({PREFIX}_KV_K(model, layer) + pos * {PREFIX}_NUM_KV_HEADS * {PREFIX}_HEAD_DIM,
               {PREFIX}_FLOAT(model, {PREFIX}_ACT.k_out),
               {PREFIX}_NUM_KV_HEADS * {PREFIX}_HEAD_DIM * sizeof(float));
        memcpy({PREFIX}_KV_V(model, layer) + pos * {PREFIX}_NUM_KV_HEADS * {PREFIX}_HEAD_DIM,
               {PREFIX}_FLOAT(model, {PREFIX}_ACT.v_out),
               {PREFIX}_NUM_KV_HEADS * {PREFIX}_HEAD_DIM * sizeof(float));

        /* Attention */
        attention_forward_causal_head_major_gqa_flash_strided(
            {PREFIX}_FLOAT(model, {PREFIX}_ACT.q_out),
            {PREFIX}_KV_K(model, layer),
            {PREFIX}_KV_V(model, layer),
            {PREFIX}_FLOAT(model, {PREFIX}_ACT.attn_out),
            {PREFIX}_NUM_HEADS, {PREFIX}_NUM_KV_HEADS, pos + 1,
            {PREFIX}_HEAD_DIM, {PREFIX}_HEAD_DIM, {PREFIX}_MAX_SEQ_LEN
        );

        /* Output projection */
        gemv_{wo_dtype}(
            {PREFIX}_ACT_A(model),
            {PREFIX}_WEIGHT(model, L->wo),
            {PREFIX}_FLOAT(model, {PREFIX}_ACT.attn_out),
            {PREFIX}_EMBED_DIM,
            {PREFIX}_NUM_HEADS * {PREFIX}_HEAD_DIM
        );

        /* Residual add */
        ck_residual_add_token_major(
            {PREFIX}_FLOAT(model, {PREFIX}_ACT.residual),
            {PREFIX}_ACT_A(model),
            {PREFIX}_ACT_A(model),
            1, {PREFIX}_EMBED_DIM
        );

        /* MLP block similar pattern... */
        // ...
    }

    /* 3. Final layer norm */
    rmsnorm_forward(
        {PREFIX}_ACT_A(model),
        {PREFIX}_FLOAT(model, {PREFIX}_HEADER.final_ln_weight),
        {PREFIX}_ACT_B(model),
        NULL, 1, {PREFIX}_EMBED_DIM, {PREFIX}_EMBED_DIM, 1e-5f
    );

    /* 4. Logits projection (weight-tied with embedding) */
    gemv_q8_0_q8_0(
        {PREFIX}_LOGITS_PTR(model),
        {PREFIX}_WEIGHT(model, {PREFIX}_HEADER.token_emb),  /* Weight tying! */
        {PREFIX}_ACT_B(model),
        {PREFIX}_VOCAB_SIZE,  /* Output dimension */
        {PREFIX}_EMBED_DIM    /* Input dimension */
    );

    model->pos++;
}
```

---

## Phase 5: Implementation Steps

### Step 1: Create `memory_layout_v6_6.py`

New script that:
1. Reads manifest + template
2. Computes layout using algorithm above
3. Outputs `{model}_{mode}_layout.json` with all offsets

### Step 2: Create `gen_layout_header_v6_6.py`

New script that:
1. Reads layout JSON
2. Generates C header file like v6.5
3. Includes all offset structs, dtype arrays, accessor macros

### Step 3: Update `codegen_v6_6.py`

Modify to:
1. Read layout header (or embed it inline)
2. Generate single-allocation model struct
3. Use accessor macros consistently
4. Implement proper buffer ping-pong

### Step 4: Update `build_ir_v6_6.py`

Modify to:
1. Call memory_layout_v6_6.py during pipeline
2. Pass layout info to codegen
3. Validate layout against IR requirements

### Step 5: Testing

1. Unit tests for offset computation
2. Canary validation for buffer overruns
3. Numerical comparison with v6.5 output
4. Memory footprint verification

---

## Appendix A: Buffer Ping-Pong Pattern

For decode mode, we use buffer A and B for input/output alternation:

```
Op 1: embedding     -> buffer_a
Op 2: ln1           buffer_a -> buffer_b
Op 3: qkv_proj      buffer_b -> q_out, k_out, v_out
Op 4: rope          q_out, k_out (in-place)
Op 5: attention     q_out, kv_cache -> attn_out
Op 6: out_proj      attn_out -> buffer_a
Op 7: residual      residual + buffer_a -> buffer_a
Op 8: ln2           buffer_a -> buffer_b
Op 9: mlp_gate_up   buffer_b -> mlp_gate
Op 10: swiglu       mlp_gate -> mlp_gate (in-place)
Op 11: mlp_down     mlp_gate -> buffer_a
Op 12: residual     residual + buffer_a -> buffer_a
... repeat for next layer
```

---

## Appendix B: Size Calculations for Qwen2-0.5B

| Component | Size (decode) | Size (prefill 32K) |
|-----------|---------------|-------------------|
| Weights | 397 MB | 397 MB |
| Activations | 60 KB | 117 MB |
| KV Cache | 402 MB | 402 MB |
| RoPE Tables | 8 MB | 8 MB |
| Logits | 608 KB | 608 KB |
| **Total** | **~808 MB** | **~925 MB** |

Note: v6.5 shows 1.5 TB because it pre-allocates all intermediate buffers for full prefill including the massive attention scores matrix.

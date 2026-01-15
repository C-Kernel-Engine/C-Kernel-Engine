/**
 * @file ck-kernel-inference.c
 * @brief AUTO-GENERATED: qwen2_0.5b_decode Implementation (IR v6 - Explicit Unrolled)
 *
 * Generated: 2026-01-15T19:23:13.600148 UTC
 * Total Memory: 3.65 GB
 * Mode: decode
 * Layers: 24 (fully unrolled)
 *
 * Per-layer quant types:
 *   Layer 0: wq=q5_0 wk=q5_0 wv=q8_0 wo=q5_0 w1=q5_0 w2=q6_k
 *   Layer 1: wq=q5_0 wk=q5_0 wv=q8_0 wo=q5_0 w1=q5_0 w2=q6_k
 *   Layer 2: wq=q5_0 wk=q5_0 wv=q5_0 wo=q5_0 w1=q5_0 w2=q4_k
 *   ... (21 more layers)
 * 
 * ═══════════════════════════════════════════════════════════════════════════
 * MANIFEST VALIDATION (from weights_manifest.json)
 * ═══════════════════════════════════════════════════════════════════════════
 * 
 * Layer | WQ    | WK    | WV    | WO    | W1    | W2    | BQ | BK | BV | BO
 * ------|-------|-------|-------|-------|-------|-------|----|----|----|----|
 *     0 | q5_0  | q5_0  | q8_0  | q5_0  | q5_0  | q6_k  | ✓  | ✓  | ✓  | ○ 
 *     1 | q5_0  | q5_0  | q8_0  | q5_0  | q5_0  | q6_k  | ✓  | ✓  | ✓  | ○ 
 *     2 | q5_0  | q5_0  | q5_0  | q5_0  | q5_0  | q4_k  | ✓  | ✓  | ✓  | ○ 
 *     3 | q5_0  | q5_0  | q8_0  | q5_0  | q5_0  | q6_k  | ✓  | ✓  | ✓  | ○ 
 *     4 | q5_0  | q5_0  | q5_0  | q5_0  | q5_0  | q4_k  | ✓  | ✓  | ✓  | ○ 
 *     5 | q5_0  | q5_0  | q5_0  | q5_0  | q5_0  | q4_k  | ✓  | ✓  | ✓  | ○ 
 *     6 | q5_0  | q5_0  | q8_0  | q5_0  | q5_0  | q6_k  | ✓  | ✓  | ✓  | ○ 
 *     7 | q5_0  | q5_0  | q8_0  | q5_0  | q5_0  | q6_k  | ✓  | ✓  | ✓  | ○ 
 *     8 | q5_0  | q5_0  | q8_0  | q5_0  | q5_0  | q6_k  | ✓  | ✓  | ✓  | ○ 
 *     9 | q5_0  | q5_0  | q8_0  | q5_0  | q5_0  | q6_k  | ✓  | ✓  | ✓  | ○ 
 *    10 | q5_0  | q5_0  | q8_0  | q5_0  | q5_0  | q6_k  | ✓  | ✓  | ✓  | ○ 
 *    11 | q5_0  | q5_0  | q5_0  | q5_0  | q5_0  | q4_k  | ✓  | ✓  | ✓  | ○ 
 *    12 | q5_0  | q5_0  | q5_0  | q5_0  | q5_0  | q4_k  | ✓  | ✓  | ✓  | ○ 
 *    13 | q5_0  | q5_0  | q8_0  | q5_0  | q5_0  | q6_k  | ✓  | ✓  | ✓  | ○ 
 *    14 | q5_0  | q5_0  | q5_0  | q5_0  | q5_0  | q4_k  | ✓  | ✓  | ✓  | ○ 
 *    15 | q5_0  | q5_0  | q5_0  | q5_0  | q5_0  | q4_k  | ✓  | ✓  | ✓  | ○ 
 *    16 | q5_0  | q5_0  | q8_0  | q5_0  | q5_0  | q6_k  | ✓  | ✓  | ✓  | ○ 
 *    17 | q5_0  | q5_0  | q5_0  | q5_0  | q5_0  | q4_k  | ✓  | ✓  | ✓  | ○ 
 *    18 | q5_0  | q5_0  | q5_0  | q5_0  | q5_0  | q4_k  | ✓  | ✓  | ✓  | ○ 
 *    19 | q5_0  | q5_0  | q8_0  | q5_0  | q5_0  | q6_k  | ✓  | ✓  | ✓  | ○ 
 *    20 | q5_0  | q5_0  | q5_0  | q5_0  | q5_0  | q4_k  | ✓  | ✓  | ✓  | ○ 
 *    21 | q5_0  | q5_0  | q8_0  | q5_0  | q5_0  | q6_k  | ✓  | ✓  | ✓  | ○ 
 *    22 | q5_0  | q5_0  | q5_0  | q5_0  | q5_0  | q4_k  | ✓  | ✓  | ✓  | ○ 
 *    23 | q5_0  | q5_0  | q5_0  | q5_0  | q5_0  | q4_k  | ✓  | ✓  | ✓  | ○ 
 * 
 * Total manifest entries: 269
 * Attention biases: PRESENT (Qwen2-style)
 * ═══════════════════════════════════════════════════════════════════════════
 * 
 *
 * DO NOT EDIT - Regenerate with build_ir_v6.py or codegen_v6.py
 */

#define _GNU_SOURCE  /* For MAP_ANONYMOUS, MAP_HUGETLB */

#include "ck-kernel-inference.h"

#include "ckernel_engine.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

#ifdef __linux__
#include <sys/mman.h>
#endif

#if QWEN2_0_5B_DECODE_DTYPE_BYTES != 4
#error "qwen2_0.5b_decode: v6 codegen currently supports fp32 only. Use --dtype=fp32."
#endif

/* ============================================================================
 * LOCAL HELPERS (no orchestration dependency)
 * ============================================================================ */

static void qwen2_0_5b_decode_residual_add_token_major(
    const float *a,
    const float *b,
    float *out,
    int tokens,
    int aligned_embed_dim
) {
    if (!a || !b || !out) {
        return;
    }
    for (int t = 0; t < tokens; ++t) {
        const float *pa = a + (size_t)t * (size_t)aligned_embed_dim;
        const float *pb = b + (size_t)t * (size_t)aligned_embed_dim;
        float *pc = out + (size_t)t * (size_t)aligned_embed_dim;
        for (int d = 0; d < aligned_embed_dim; ++d) {
            pc[d] = pa[d] + pb[d];
        }
    }
}

/* ============================================================================
 * MAGIC HEADER
 * ============================================================================ */

typedef struct __attribute__((packed)) {
    uint32_t magic;           /* 0x434B454E */
    uint32_t version;          /* IR version */
    uint64_t total_bytes;
    uint64_t weight_bytes;
    uint64_t activation_bytes;
    uint32_t num_layers;
    uint32_t embed_dim;
    uint32_t num_heads;
    uint32_t vocab_size;
    uint32_t max_seq_len;
    uint32_t canary_count;
    uint8_t  reserved[8];       /* Pad to 64 bytes */
} MagicHeader;

_Static_assert(sizeof(MagicHeader) == 64, "MagicHeader must be 64 bytes");

/* ============================================================================
 * ALLOCATION
 * ============================================================================ */

int qwen2_0_5b_decode_model_allocate(QWEN2_0_5B_DECODEModel *model) {
    size_t total = QWEN2_0_5B_DECODE_TOTAL_BYTES;

#ifdef __linux__
    model->base = mmap(NULL, total,
                       PROT_READ | PROT_WRITE,
                       MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB,
                       -1, 0);
    if (model->base == MAP_FAILED) {
        model->base = mmap(NULL, total,
                           PROT_READ | PROT_WRITE,
                           MAP_PRIVATE | MAP_ANONYMOUS,
                           -1, 0);
    }
    if (model->base == MAP_FAILED) {
        perror("mmap failed");
        return -1;
    }
#else
    model->base = aligned_alloc(64, total);
    if (!model->base) {
        perror("aligned_alloc failed");
        return -1;
    }
#endif

    model->total_bytes = total;

    /* Initialize magic header */
    MagicHeader *header = (MagicHeader *)model->base;
    header->magic = QWEN2_0_5B_DECODE_MAGIC;
    header->version = 5;
    header->total_bytes = QWEN2_0_5B_DECODE_TOTAL_BYTES;
    header->weight_bytes = QWEN2_0_5B_DECODE_WEIGHT_BYTES;
    header->activation_bytes = QWEN2_0_5B_DECODE_ACTIVATION_BYTES;
    header->num_layers = QWEN2_0_5B_DECODE_NUM_LAYERS;
    header->embed_dim = QWEN2_0_5B_DECODE_EMBED_DIM;
    header->num_heads = QWEN2_0_5B_DECODE_NUM_HEADS;
    header->vocab_size = QWEN2_0_5B_DECODE_VOCAB_SIZE;
    header->max_seq_len = QWEN2_0_5B_DECODE_MAX_SEQ_LEN;
    header->canary_count = QWEN2_0_5B_DECODE_CANARY_COUNT;

    /* Initialize canary guards */
    for (int i = 0; i < QWEN2_0_5B_DECODE_CANARY_COUNT; i++) {
        uint32_t *ptr = (uint32_t*)((char*)model->base + QWEN2_0_5B_DECODE_CANARIES[i].offset);
        for (int j = 0; j < (QWEN2_0_5B_DECODE_CANARY_SIZE / 4); j++) {
            ptr[j] = QWEN2_0_5B_DECODE_CANARY_VALUE;
        }
    }

    return 0;
}

void qwen2_0_5b_decode_model_free(QWEN2_0_5B_DECODEModel *model) {
    if (!model || !model->base) return;
#ifdef __linux__
    munmap(model->base, model->total_bytes);
#else
    free(model->base);
#endif
    model->base = NULL;
    model->total_bytes = 0;
}

int qwen2_0_5b_decode_verify_canaries(QWEN2_0_5B_DECODEModel *model) {
    int errors = 0;
    uint32_t *ptr;

    for (int i = 0; i < QWEN2_0_5B_DECODE_CANARY_COUNT; i++) {
        ptr = (uint32_t*)((char*)model->base + QWEN2_0_5B_DECODE_CANARIES[i].offset);
        for (int j = 0; j < 4; j++) {
            if (ptr[j] != QWEN2_0_5B_DECODE_CANARY_VALUE) {
                fprintf(stderr, "CANARY CORRUPTION: %s at offset 0x%lX\n",
                        QWEN2_0_5B_DECODE_CANARIES[i].name,
                        QWEN2_0_5B_DECODE_CANARIES[i].offset);
                errors++;
                break;
            }
        }
    }

    return errors;
}

/* ============================================================================
 * ALIGNMENT HELPERS
 * ============================================================================ */

static int qwen2_0_5b_decode_align_elems(int elems, int elem_bytes, int align_bytes) {
    int bytes = elems * elem_bytes;
    int aligned = (bytes + align_bytes - 1) / align_bytes * align_bytes;
    return aligned / elem_bytes;
}

/* ============================================================================
 * ROPE PRECOMPUTE
 * ============================================================================ */

void qwen2_0_5b_decode_precompute_rope(QWEN2_0_5B_DECODEModel *model) {
    const int T = QWEN2_0_5B_DECODE_MAX_SEQ_LEN;
    const int D = QWEN2_0_5B_DECODE_HEAD_DIM / 2;
    const float theta = 1000000.0f;

    float *cos_ptr = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_GLOBALS.rope_cos_cache);
    float *sin_ptr = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_GLOBALS.rope_sin_cache);

    for (int pos = 0; pos < T; pos++) {
        for (int i = 0; i < D; i++) {
            float freq = 1.0f / powf(theta, (float)(2 * i) / (float)(D * 2));
            float angle = (float)pos * freq;
            cos_ptr[pos * D + i] = cosf(angle);
            sin_ptr[pos * D + i] = sinf(angle);
        }
    }
}

/* ============================================================================
 * EXPLICIT PER-LAYER PREFILL FUNCTIONS (v6 unrolled)
 * ============================================================================ */

/*
 * Layer 0: wq=q5_0 wk=q5_0 wv=q8_0 wo=q5_0 w1=q5_0 w2=q6_k
 */

static void qwen2_0_5b_decode_forward_prefill_impl(
    QWEN2_0_5B_DECODEModel *model,
    const int *tokens,
    int num_tokens
) {
    if (!model || !tokens || num_tokens <= 0) {
        return;
    }

    const int elem_bytes = QWEN2_0_5B_DECODE_DTYPE_BYTES;
    const int aligned_embed_dim = 896;
    const int aligned_head_dim = 64;
    const int aligned_intermediate_dim = 4864;
    const int aligned_context_window = 131072;

    float *embed_out = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_HEADER.embedded_input);
    const void *embed_weight = (const void *)QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_HEADER.token_emb);
    embedding_forward_q8_0((const int32_t *)tokens,
                          num_tokens,
                          QWEN2_0_5B_DECODE_VOCAB_SIZE,
                          embed_weight,
                          NULL,
                          embed_out,
                          QWEN2_0_5B_DECODE_EMBED_DIM,
                          aligned_embed_dim,
                          num_tokens,
                          0);

    qwen2_0_5b_decode_layer_0_prefill(
        model,
        num_tokens,
        aligned_embed_dim,
        aligned_head_dim,
        aligned_intermediate_dim,
        aligned_context_window);

    kv_cache_repack_head_major_inplace(
        QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_LAYERS[0].k),
        QWEN2_0_5B_DECODE_NUM_KV_HEADS,
        num_tokens,
        aligned_context_window,
        aligned_head_dim);
    kv_cache_repack_head_major_inplace(
        QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_LAYERS[0].v),
        QWEN2_0_5B_DECODE_NUM_KV_HEADS,
        num_tokens,
        aligned_context_window,
        aligned_head_dim);

    qwen2_0_5b_decode_layer_1_prefill(
        model,
        num_tokens,
        aligned_embed_dim,
        aligned_head_dim,
        aligned_intermediate_dim,
        aligned_context_window);

    kv_cache_repack_head_major_inplace(
        QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_LAYERS[1].k),
        QWEN2_0_5B_DECODE_NUM_KV_HEADS,
        num_tokens,
        aligned_context_window,
        aligned_head_dim);
    kv_cache_repack_head_major_inplace(
        QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_LAYERS[1].v),
        QWEN2_0_5B_DECODE_NUM_KV_HEADS,
        num_tokens,
        aligned_context_window,
        aligned_head_dim);

    qwen2_0_5b_decode_layer_2_prefill(
        model,
        num_tokens,
        aligned_embed_dim,
        aligned_head_dim,
        aligned_intermediate_dim,
        aligned_context_window);

    kv_cache_repack_head_major_inplace(
        QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_LAYERS[2].k),
        QWEN2_0_5B_DECODE_NUM_KV_HEADS,
        num_tokens,
        aligned_context_window,
        aligned_head_dim);
    kv_cache_repack_head_major_inplace(
        QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_LAYERS[2].v),
        QWEN2_0_5B_DECODE_NUM_KV_HEADS,
        num_tokens,
        aligned_context_window,
        aligned_head_dim);

    qwen2_0_5b_decode_layer_3_prefill(
        model,
        num_tokens,
        aligned_embed_dim,
        aligned_head_dim,
        aligned_intermediate_dim,
        aligned_context_window);

    kv_cache_repack_head_major_inplace(
        QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_LAYERS[3].k),
        QWEN2_0_5B_DECODE_NUM_KV_HEADS,
        num_tokens,
        aligned_context_window,
        aligned_head_dim);
    kv_cache_repack_head_major_inplace(
        QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_LAYERS[3].v),
        QWEN2_0_5B_DECODE_NUM_KV_HEADS,
        num_tokens,
        aligned_context_window,
        aligned_head_dim);

    qwen2_0_5b_decode_layer_4_prefill(
        model,
        num_tokens,
        aligned_embed_dim,
        aligned_head_dim,
        aligned_intermediate_dim,
        aligned_context_window);

    kv_cache_repack_head_major_inplace(
        QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_LAYERS[4].k),
        QWEN2_0_5B_DECODE_NUM_KV_HEADS,
        num_tokens,
        aligned_context_window,
        aligned_head_dim);
    kv_cache_repack_head_major_inplace(
        QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_LAYERS[4].v),
        QWEN2_0_5B_DECODE_NUM_KV_HEADS,
        num_tokens,
        aligned_context_window,
        aligned_head_dim);

    qwen2_0_5b_decode_layer_5_prefill(
        model,
        num_tokens,
        aligned_embed_dim,
        aligned_head_dim,
        aligned_intermediate_dim,
        aligned_context_window);

    kv_cache_repack_head_major_inplace(
        QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_LAYERS[5].k),
        QWEN2_0_5B_DECODE_NUM_KV_HEADS,
        num_tokens,
        aligned_context_window,
        aligned_head_dim);
    kv_cache_repack_head_major_inplace(
        QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_LAYERS[5].v),
        QWEN2_0_5B_DECODE_NUM_KV_HEADS,
        num_tokens,
        aligned_context_window,
        aligned_head_dim);

    qwen2_0_5b_decode_layer_6_prefill(
        model,
        num_tokens,
        aligned_embed_dim,
        aligned_head_dim,
        aligned_intermediate_dim,
        aligned_context_window);

    kv_cache_repack_head_major_inplace(
        QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_LAYERS[6].k),
        QWEN2_0_5B_DECODE_NUM_KV_HEADS,
        num_tokens,
        aligned_context_window,
        aligned_head_dim);
    kv_cache_repack_head_major_inplace(
        QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_LAYERS[6].v),
        QWEN2_0_5B_DECODE_NUM_KV_HEADS,
        num_tokens,
        aligned_context_window,
        aligned_head_dim);

    qwen2_0_5b_decode_layer_7_prefill(
        model,
        num_tokens,
        aligned_embed_dim,
        aligned_head_dim,
        aligned_intermediate_dim,
        aligned_context_window);

    kv_cache_repack_head_major_inplace(
        QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_LAYERS[7].k),
        QWEN2_0_5B_DECODE_NUM_KV_HEADS,
        num_tokens,
        aligned_context_window,
        aligned_head_dim);
    kv_cache_repack_head_major_inplace(
        QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_LAYERS[7].v),
        QWEN2_0_5B_DECODE_NUM_KV_HEADS,
        num_tokens,
        aligned_context_window,
        aligned_head_dim);

    qwen2_0_5b_decode_layer_8_prefill(
        model,
        num_tokens,
        aligned_embed_dim,
        aligned_head_dim,
        aligned_intermediate_dim,
        aligned_context_window);

    kv_cache_repack_head_major_inplace(
        QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_LAYERS[8].k),
        QWEN2_0_5B_DECODE_NUM_KV_HEADS,
        num_tokens,
        aligned_context_window,
        aligned_head_dim);
    kv_cache_repack_head_major_inplace(
        QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_LAYERS[8].v),
        QWEN2_0_5B_DECODE_NUM_KV_HEADS,
        num_tokens,
        aligned_context_window,
        aligned_head_dim);

    qwen2_0_5b_decode_layer_9_prefill(
        model,
        num_tokens,
        aligned_embed_dim,
        aligned_head_dim,
        aligned_intermediate_dim,
        aligned_context_window);

    kv_cache_repack_head_major_inplace(
        QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_LAYERS[9].k),
        QWEN2_0_5B_DECODE_NUM_KV_HEADS,
        num_tokens,
        aligned_context_window,
        aligned_head_dim);
    kv_cache_repack_head_major_inplace(
        QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_LAYERS[9].v),
        QWEN2_0_5B_DECODE_NUM_KV_HEADS,
        num_tokens,
        aligned_context_window,
        aligned_head_dim);

    qwen2_0_5b_decode_layer_10_prefill(
        model,
        num_tokens,
        aligned_embed_dim,
        aligned_head_dim,
        aligned_intermediate_dim,
        aligned_context_window);

    kv_cache_repack_head_major_inplace(
        QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_LAYERS[10].k),
        QWEN2_0_5B_DECODE_NUM_KV_HEADS,
        num_tokens,
        aligned_context_window,
        aligned_head_dim);
    kv_cache_repack_head_major_inplace(
        QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_LAYERS[10].v),
        QWEN2_0_5B_DECODE_NUM_KV_HEADS,
        num_tokens,
        aligned_context_window,
        aligned_head_dim);

    qwen2_0_5b_decode_layer_11_prefill(
        model,
        num_tokens,
        aligned_embed_dim,
        aligned_head_dim,
        aligned_intermediate_dim,
        aligned_context_window);

    kv_cache_repack_head_major_inplace(
        QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_LAYERS[11].k),
        QWEN2_0_5B_DECODE_NUM_KV_HEADS,
        num_tokens,
        aligned_context_window,
        aligned_head_dim);
    kv_cache_repack_head_major_inplace(
        QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_LAYERS[11].v),
        QWEN2_0_5B_DECODE_NUM_KV_HEADS,
        num_tokens,
        aligned_context_window,
        aligned_head_dim);

    qwen2_0_5b_decode_layer_12_prefill(
        model,
        num_tokens,
        aligned_embed_dim,
        aligned_head_dim,
        aligned_intermediate_dim,
        aligned_context_window);

    kv_cache_repack_head_major_inplace(
        QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_LAYERS[12].k),
        QWEN2_0_5B_DECODE_NUM_KV_HEADS,
        num_tokens,
        aligned_context_window,
        aligned_head_dim);
    kv_cache_repack_head_major_inplace(
        QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_LAYERS[12].v),
        QWEN2_0_5B_DECODE_NUM_KV_HEADS,
        num_tokens,
        aligned_context_window,
        aligned_head_dim);

    qwen2_0_5b_decode_layer_13_prefill(
        model,
        num_tokens,
        aligned_embed_dim,
        aligned_head_dim,
        aligned_intermediate_dim,
        aligned_context_window);

    kv_cache_repack_head_major_inplace(
        QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_LAYERS[13].k),
        QWEN2_0_5B_DECODE_NUM_KV_HEADS,
        num_tokens,
        aligned_context_window,
        aligned_head_dim);
    kv_cache_repack_head_major_inplace(
        QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_LAYERS[13].v),
        QWEN2_0_5B_DECODE_NUM_KV_HEADS,
        num_tokens,
        aligned_context_window,
        aligned_head_dim);

    qwen2_0_5b_decode_layer_14_prefill(
        model,
        num_tokens,
        aligned_embed_dim,
        aligned_head_dim,
        aligned_intermediate_dim,
        aligned_context_window);

    kv_cache_repack_head_major_inplace(
        QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_LAYERS[14].k),
        QWEN2_0_5B_DECODE_NUM_KV_HEADS,
        num_tokens,
        aligned_context_window,
        aligned_head_dim);
    kv_cache_repack_head_major_inplace(
        QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_LAYERS[14].v),
        QWEN2_0_5B_DECODE_NUM_KV_HEADS,
        num_tokens,
        aligned_context_window,
        aligned_head_dim);

    qwen2_0_5b_decode_layer_15_prefill(
        model,
        num_tokens,
        aligned_embed_dim,
        aligned_head_dim,
        aligned_intermediate_dim,
        aligned_context_window);

    kv_cache_repack_head_major_inplace(
        QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_LAYERS[15].k),
        QWEN2_0_5B_DECODE_NUM_KV_HEADS,
        num_tokens,
        aligned_context_window,
        aligned_head_dim);
    kv_cache_repack_head_major_inplace(
        QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_LAYERS[15].v),
        QWEN2_0_5B_DECODE_NUM_KV_HEADS,
        num_tokens,
        aligned_context_window,
        aligned_head_dim);

    qwen2_0_5b_decode_layer_16_prefill(
        model,
        num_tokens,
        aligned_embed_dim,
        aligned_head_dim,
        aligned_intermediate_dim,
        aligned_context_window);

    kv_cache_repack_head_major_inplace(
        QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_LAYERS[16].k),
        QWEN2_0_5B_DECODE_NUM_KV_HEADS,
        num_tokens,
        aligned_context_window,
        aligned_head_dim);
    kv_cache_repack_head_major_inplace(
        QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_LAYERS[16].v),
        QWEN2_0_5B_DECODE_NUM_KV_HEADS,
        num_tokens,
        aligned_context_window,
        aligned_head_dim);

    qwen2_0_5b_decode_layer_17_prefill(
        model,
        num_tokens,
        aligned_embed_dim,
        aligned_head_dim,
        aligned_intermediate_dim,
        aligned_context_window);

    kv_cache_repack_head_major_inplace(
        QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_LAYERS[17].k),
        QWEN2_0_5B_DECODE_NUM_KV_HEADS,
        num_tokens,
        aligned_context_window,
        aligned_head_dim);
    kv_cache_repack_head_major_inplace(
        QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_LAYERS[17].v),
        QWEN2_0_5B_DECODE_NUM_KV_HEADS,
        num_tokens,
        aligned_context_window,
        aligned_head_dim);

    qwen2_0_5b_decode_layer_18_prefill(
        model,
        num_tokens,
        aligned_embed_dim,
        aligned_head_dim,
        aligned_intermediate_dim,
        aligned_context_window);

    kv_cache_repack_head_major_inplace(
        QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_LAYERS[18].k),
        QWEN2_0_5B_DECODE_NUM_KV_HEADS,
        num_tokens,
        aligned_context_window,
        aligned_head_dim);
    kv_cache_repack_head_major_inplace(
        QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_LAYERS[18].v),
        QWEN2_0_5B_DECODE_NUM_KV_HEADS,
        num_tokens,
        aligned_context_window,
        aligned_head_dim);

    qwen2_0_5b_decode_layer_19_prefill(
        model,
        num_tokens,
        aligned_embed_dim,
        aligned_head_dim,
        aligned_intermediate_dim,
        aligned_context_window);

    kv_cache_repack_head_major_inplace(
        QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_LAYERS[19].k),
        QWEN2_0_5B_DECODE_NUM_KV_HEADS,
        num_tokens,
        aligned_context_window,
        aligned_head_dim);
    kv_cache_repack_head_major_inplace(
        QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_LAYERS[19].v),
        QWEN2_0_5B_DECODE_NUM_KV_HEADS,
        num_tokens,
        aligned_context_window,
        aligned_head_dim);

    qwen2_0_5b_decode_layer_20_prefill(
        model,
        num_tokens,
        aligned_embed_dim,
        aligned_head_dim,
        aligned_intermediate_dim,
        aligned_context_window);

    kv_cache_repack_head_major_inplace(
        QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_LAYERS[20].k),
        QWEN2_0_5B_DECODE_NUM_KV_HEADS,
        num_tokens,
        aligned_context_window,
        aligned_head_dim);
    kv_cache_repack_head_major_inplace(
        QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_LAYERS[20].v),
        QWEN2_0_5B_DECODE_NUM_KV_HEADS,
        num_tokens,
        aligned_context_window,
        aligned_head_dim);

    qwen2_0_5b_decode_layer_21_prefill(
        model,
        num_tokens,
        aligned_embed_dim,
        aligned_head_dim,
        aligned_intermediate_dim,
        aligned_context_window);

    kv_cache_repack_head_major_inplace(
        QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_LAYERS[21].k),
        QWEN2_0_5B_DECODE_NUM_KV_HEADS,
        num_tokens,
        aligned_context_window,
        aligned_head_dim);
    kv_cache_repack_head_major_inplace(
        QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_LAYERS[21].v),
        QWEN2_0_5B_DECODE_NUM_KV_HEADS,
        num_tokens,
        aligned_context_window,
        aligned_head_dim);

    qwen2_0_5b_decode_layer_22_prefill(
        model,
        num_tokens,
        aligned_embed_dim,
        aligned_head_dim,
        aligned_intermediate_dim,
        aligned_context_window);

    kv_cache_repack_head_major_inplace(
        QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_LAYERS[22].k),
        QWEN2_0_5B_DECODE_NUM_KV_HEADS,
        num_tokens,
        aligned_context_window,
        aligned_head_dim);
    kv_cache_repack_head_major_inplace(
        QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_LAYERS[22].v),
        QWEN2_0_5B_DECODE_NUM_KV_HEADS,
        num_tokens,
        aligned_context_window,
        aligned_head_dim);

    qwen2_0_5b_decode_layer_23_prefill(
        model,
        num_tokens,
        aligned_embed_dim,
        aligned_head_dim,
        aligned_intermediate_dim,
        aligned_context_window);

    kv_cache_repack_head_major_inplace(
        QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_LAYERS[23].k),
        QWEN2_0_5B_DECODE_NUM_KV_HEADS,
        num_tokens,
        aligned_context_window,
        aligned_head_dim);
    kv_cache_repack_head_major_inplace(
        QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_LAYERS[23].v),
        QWEN2_0_5B_DECODE_NUM_KV_HEADS,
        num_tokens,
        aligned_context_window,
        aligned_head_dim);

    float *last_hidden = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_LAYERS[QWEN2_0_5B_DECODE_NUM_LAYERS - 1].output);
    float *final_ln_weight = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_FOOTER.final_ln_weight);
    float *final_out = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_FOOTER.final_output);
    rmsnorm_forward(last_hidden,
                   final_ln_weight,
                   final_out,
                   NULL,
                   num_tokens,
                   QWEN2_0_5B_DECODE_EMBED_DIM,
                   aligned_embed_dim,
                   1e-06f);

    float *logits = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_FOOTER.logits);
    const void *lm_head = (const void *)QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_FOOTER.lm_head_weight);
    for (int t = 0; t < num_tokens; ++t) {
        const float *row = final_out + (size_t)t * (size_t)aligned_embed_dim;
        float *logits_row = logits + (size_t)t * (size_t)QWEN2_0_5B_DECODE_VOCAB_SIZE;
        gemm_nt_q8_0(row,
                       lm_head,
                       NULL,
                       logits_row,
                       1,
                       QWEN2_0_5B_DECODE_VOCAB_SIZE,
                       aligned_embed_dim);
    }
}

/* ============================================================================
 * EXPLICIT PER-LAYER DECODE FUNCTIONS (v6 unrolled)
 * ============================================================================ */

/*
 * Layer 0: wq=q5_0 wk=q5_0 wv=q8_0 wo=q5_0 w1=q5_0 w2=q6_k
 */
static void qwen2_0_5b_decode_layer_0_decode(
    QWEN2_0_5B_DECODEModel *model,
    int token_index,
    int aligned_embed_dim,
    int aligned_head_dim,
    int aligned_intermediate_dim,
    int aligned_context_window
) {
    const QWEN2_0_5B_DECODELayerOffsets *L = &QWEN2_0_5B_DECODE_LAYERS[0];

    float *input = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_HEADER.embedded_input);

    float *ln1_gamma = QWEN2_0_5B_DECODE_PTR(model, L->ln1_gamma);
    float *ln1_out = QWEN2_0_5B_DECODE_PTR(model, L->ln1_out);
    float *ln2_gamma = QWEN2_0_5B_DECODE_PTR(model, L->ln2_gamma);
    float *ln2_out = QWEN2_0_5B_DECODE_PTR(model, L->ln2_out);
    float *k_cache = QWEN2_0_5B_DECODE_PTR(model, L->k);
    float *v_cache = QWEN2_0_5B_DECODE_PTR(model, L->v);
    float *proj_tmp = QWEN2_0_5B_DECODE_PTR(model, L->proj_tmp);
    float *proj_scratch = QWEN2_0_5B_DECODE_PTR(model, L->proj_scratch);
    float *residual1 = QWEN2_0_5B_DECODE_PTR(model, L->residual1);
    float *mlp_out = QWEN2_0_5B_DECODE_PTR(model, L->mlp_out);
    float *output = QWEN2_0_5B_DECODE_PTR(model, L->output);

    /* Weights (explicit types for layer 0) */
    const void *WQ = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wq);  /* Q5_0 */
    const void *WK = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wk);  /* Q5_0 */
    const void *WV = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wv);  /* Q8_0 */
    const void *WO = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wo);  /* Q5_0 */
    const void *W1 = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->w1);  /* Q5_0 (gate+up) */
    const void *W2 = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->w2);  /* Q6_K (down) */

    /* Attention biases (Qwen2-style) */
    const float *BQ = (const float *)QWEN2_0_5B_DECODE_PTR(model, L->bq);
    const float *BK = (const float *)QWEN2_0_5B_DECODE_PTR(model, L->bk);
    const float *BV = (const float *)QWEN2_0_5B_DECODE_PTR(model, L->bv);

    float *rope_cos = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_GLOBALS.rope_cos_cache);
    float *rope_sin = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_GLOBALS.rope_sin_cache);

    const int H = QWEN2_0_5B_DECODE_NUM_HEADS;
    const int H_kv = QWEN2_0_5B_DECODE_NUM_KV_HEADS;
    const int head_dim = QWEN2_0_5B_DECODE_HEAD_DIM;

    float q_token[H * aligned_head_dim];
    float k_token[H_kv * aligned_head_dim];
    float v_token[H_kv * aligned_head_dim];
    float attn_token[H * aligned_head_dim];

    /* Local MLP buffers (avoid layout dependencies for intermediate values) */
    float fc1_out[2 * aligned_intermediate_dim];
    float swiglu_out[aligned_intermediate_dim];

    /* Step 1: RMSNorm before attention */
    rmsnorm_forward(input,
                    ln1_gamma,
                    ln1_out,
                    NULL,
                    1,
                    QWEN2_0_5B_DECODE_EMBED_DIM,
                    aligned_embed_dim,
                    1e-06f);

    /* Step 2: QKV projection */
    /* Q projection: Q5_0 -> gemm_nt_q5_0 */
    gemm_nt_q5_0(ln1_out, WQ, BQ, q_token, 1, H * head_dim, aligned_embed_dim);

    /* K projection: Q5_0 -> gemm_nt_q5_0 */
    gemm_nt_q5_0(ln1_out, WK, BK, k_token, 1, H_kv * head_dim, aligned_embed_dim);

    /* V projection: Q8_0 -> gemm_nt_q8_0 */
    gemm_nt_q8_0(ln1_out, WV, BV, v_token, 1, H_kv * head_dim, aligned_embed_dim);

    /* Step 3: RoPE */
    rope_forward_qk(q_token,
                    k_token,
                    rope_cos,
                    rope_sin,
                    H,
                    H_kv,
                    1,
                    head_dim,
                    aligned_head_dim,
                    token_index);

    /* Step 4: KV cache write */
    kv_cache_write_head_major(k_token,
                              v_token,
                              k_cache,
                              v_cache,
                              H_kv,
                              token_index,
                              aligned_context_window,
                              head_dim,
                              aligned_head_dim);

    /* Step 5: Attention (decode) */
    attention_forward_decode_head_major_gqa_regular(q_token,
                                                   k_cache,
                                                   v_cache,
                                                   attn_token,
                                                   H,
                                                   H_kv,
                                                   token_index + 1,
                                                   aligned_context_window,
                                                   head_dim,
                                                   aligned_head_dim);

    /* Step 6: Output projection */
    /* WO projection: Q5_0 -> gemm_nt_q5_0 */
    gemm_nt_q5_0(attn_token, WO, NULL, proj_tmp, 1, aligned_embed_dim, H * head_dim);

    /* Step 7: Residual add */
    qwen2_0_5b_decode_residual_add_token_major(input, proj_tmp, residual1, 1, aligned_embed_dim);

    /* Step 8: RMSNorm before MLP */
    rmsnorm_forward(residual1,
                    ln2_gamma,
                    ln2_out,
                    NULL,
                    1,
                    QWEN2_0_5B_DECODE_EMBED_DIM,
                    aligned_embed_dim,
                    1e-06f);

    /* Step 9: MLP (SwiGLU) */
    /* Gate+Up projection: Q5_0 -> gemm_nt_q5_0 */
    gemm_nt_q5_0(ln2_out, W1, NULL, fc1_out, 1, 2 * aligned_intermediate_dim, aligned_embed_dim);

    /* SwiGLU activation */
    swiglu_forward(fc1_out, swiglu_out, 1, aligned_intermediate_dim);

    /* Down projection: Q6_K -> gemm_nt_q6_k */
    gemm_nt_q6_k(swiglu_out, W2, NULL, mlp_out, 1, aligned_embed_dim, aligned_intermediate_dim);

    /* Step 10: Final residual add */
    qwen2_0_5b_decode_residual_add_token_major(residual1, mlp_out, output, 1, aligned_embed_dim);
}

/*
 * Layer 1: wq=q5_0 wk=q5_0 wv=q8_0 wo=q5_0 w1=q5_0 w2=q6_k
 */
static void qwen2_0_5b_decode_layer_1_decode(
    QWEN2_0_5B_DECODEModel *model,
    int token_index,
    int aligned_embed_dim,
    int aligned_head_dim,
    int aligned_intermediate_dim,
    int aligned_context_window
) {
    const QWEN2_0_5B_DECODELayerOffsets *L = &QWEN2_0_5B_DECODE_LAYERS[1];

    float *input = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_LAYERS[0].output);

    float *ln1_gamma = QWEN2_0_5B_DECODE_PTR(model, L->ln1_gamma);
    float *ln1_out = QWEN2_0_5B_DECODE_PTR(model, L->ln1_out);
    float *ln2_gamma = QWEN2_0_5B_DECODE_PTR(model, L->ln2_gamma);
    float *ln2_out = QWEN2_0_5B_DECODE_PTR(model, L->ln2_out);
    float *k_cache = QWEN2_0_5B_DECODE_PTR(model, L->k);
    float *v_cache = QWEN2_0_5B_DECODE_PTR(model, L->v);
    float *proj_tmp = QWEN2_0_5B_DECODE_PTR(model, L->proj_tmp);
    float *proj_scratch = QWEN2_0_5B_DECODE_PTR(model, L->proj_scratch);
    float *residual1 = QWEN2_0_5B_DECODE_PTR(model, L->residual1);
    float *mlp_out = QWEN2_0_5B_DECODE_PTR(model, L->mlp_out);
    float *output = QWEN2_0_5B_DECODE_PTR(model, L->output);

    /* Weights (explicit types for layer 1) */
    const void *WQ = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wq);  /* Q5_0 */
    const void *WK = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wk);  /* Q5_0 */
    const void *WV = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wv);  /* Q8_0 */
    const void *WO = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wo);  /* Q5_0 */
    const void *W1 = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->w1);  /* Q5_0 (gate+up) */
    const void *W2 = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->w2);  /* Q6_K (down) */

    /* Attention biases (Qwen2-style) */
    const float *BQ = (const float *)QWEN2_0_5B_DECODE_PTR(model, L->bq);
    const float *BK = (const float *)QWEN2_0_5B_DECODE_PTR(model, L->bk);
    const float *BV = (const float *)QWEN2_0_5B_DECODE_PTR(model, L->bv);

    float *rope_cos = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_GLOBALS.rope_cos_cache);
    float *rope_sin = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_GLOBALS.rope_sin_cache);

    const int H = QWEN2_0_5B_DECODE_NUM_HEADS;
    const int H_kv = QWEN2_0_5B_DECODE_NUM_KV_HEADS;
    const int head_dim = QWEN2_0_5B_DECODE_HEAD_DIM;

    float q_token[H * aligned_head_dim];
    float k_token[H_kv * aligned_head_dim];
    float v_token[H_kv * aligned_head_dim];
    float attn_token[H * aligned_head_dim];

    /* Local MLP buffers (avoid layout dependencies for intermediate values) */
    float fc1_out[2 * aligned_intermediate_dim];
    float swiglu_out[aligned_intermediate_dim];

    /* Step 1: RMSNorm before attention */
    rmsnorm_forward(input,
                    ln1_gamma,
                    ln1_out,
                    NULL,
                    1,
                    QWEN2_0_5B_DECODE_EMBED_DIM,
                    aligned_embed_dim,
                    1e-06f);

    /* Step 2: QKV projection */
    /* Q projection: Q5_0 -> gemm_nt_q5_0 */
    gemm_nt_q5_0(ln1_out, WQ, BQ, q_token, 1, H * head_dim, aligned_embed_dim);

    /* K projection: Q5_0 -> gemm_nt_q5_0 */
    gemm_nt_q5_0(ln1_out, WK, BK, k_token, 1, H_kv * head_dim, aligned_embed_dim);

    /* V projection: Q8_0 -> gemm_nt_q8_0 */
    gemm_nt_q8_0(ln1_out, WV, BV, v_token, 1, H_kv * head_dim, aligned_embed_dim);

    /* Step 3: RoPE */
    rope_forward_qk(q_token,
                    k_token,
                    rope_cos,
                    rope_sin,
                    H,
                    H_kv,
                    1,
                    head_dim,
                    aligned_head_dim,
                    token_index);

    /* Step 4: KV cache write */
    kv_cache_write_head_major(k_token,
                              v_token,
                              k_cache,
                              v_cache,
                              H_kv,
                              token_index,
                              aligned_context_window,
                              head_dim,
                              aligned_head_dim);

    /* Step 5: Attention (decode) */
    attention_forward_decode_head_major_gqa_regular(q_token,
                                                   k_cache,
                                                   v_cache,
                                                   attn_token,
                                                   H,
                                                   H_kv,
                                                   token_index + 1,
                                                   aligned_context_window,
                                                   head_dim,
                                                   aligned_head_dim);

    /* Step 6: Output projection */
    /* WO projection: Q5_0 -> gemm_nt_q5_0 */
    gemm_nt_q5_0(attn_token, WO, NULL, proj_tmp, 1, aligned_embed_dim, H * head_dim);

    /* Step 7: Residual add */
    qwen2_0_5b_decode_residual_add_token_major(input, proj_tmp, residual1, 1, aligned_embed_dim);

    /* Step 8: RMSNorm before MLP */
    rmsnorm_forward(residual1,
                    ln2_gamma,
                    ln2_out,
                    NULL,
                    1,
                    QWEN2_0_5B_DECODE_EMBED_DIM,
                    aligned_embed_dim,
                    1e-06f);

    /* Step 9: MLP (SwiGLU) */
    /* Gate+Up projection: Q5_0 -> gemm_nt_q5_0 */
    gemm_nt_q5_0(ln2_out, W1, NULL, fc1_out, 1, 2 * aligned_intermediate_dim, aligned_embed_dim);

    /* SwiGLU activation */
    swiglu_forward(fc1_out, swiglu_out, 1, aligned_intermediate_dim);

    /* Down projection: Q6_K -> gemm_nt_q6_k */
    gemm_nt_q6_k(swiglu_out, W2, NULL, mlp_out, 1, aligned_embed_dim, aligned_intermediate_dim);

    /* Step 10: Final residual add */
    qwen2_0_5b_decode_residual_add_token_major(residual1, mlp_out, output, 1, aligned_embed_dim);
}

/*
 * Layer 2: wq=q5_0 wk=q5_0 wv=q5_0 wo=q5_0 w1=q5_0 w2=q4_k
 */
static void qwen2_0_5b_decode_layer_2_decode(
    QWEN2_0_5B_DECODEModel *model,
    int token_index,
    int aligned_embed_dim,
    int aligned_head_dim,
    int aligned_intermediate_dim,
    int aligned_context_window
) {
    const QWEN2_0_5B_DECODELayerOffsets *L = &QWEN2_0_5B_DECODE_LAYERS[2];

    float *input = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_LAYERS[1].output);

    float *ln1_gamma = QWEN2_0_5B_DECODE_PTR(model, L->ln1_gamma);
    float *ln1_out = QWEN2_0_5B_DECODE_PTR(model, L->ln1_out);
    float *ln2_gamma = QWEN2_0_5B_DECODE_PTR(model, L->ln2_gamma);
    float *ln2_out = QWEN2_0_5B_DECODE_PTR(model, L->ln2_out);
    float *k_cache = QWEN2_0_5B_DECODE_PTR(model, L->k);
    float *v_cache = QWEN2_0_5B_DECODE_PTR(model, L->v);
    float *proj_tmp = QWEN2_0_5B_DECODE_PTR(model, L->proj_tmp);
    float *proj_scratch = QWEN2_0_5B_DECODE_PTR(model, L->proj_scratch);
    float *residual1 = QWEN2_0_5B_DECODE_PTR(model, L->residual1);
    float *mlp_out = QWEN2_0_5B_DECODE_PTR(model, L->mlp_out);
    float *output = QWEN2_0_5B_DECODE_PTR(model, L->output);

    /* Weights (explicit types for layer 2) */
    const void *WQ = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wq);  /* Q5_0 */
    const void *WK = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wk);  /* Q5_0 */
    const void *WV = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wv);  /* Q5_0 */
    const void *WO = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wo);  /* Q5_0 */
    const void *W1 = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->w1);  /* Q5_0 (gate+up) */
    const void *W2 = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->w2);  /* Q4_K (down) */

    /* Attention biases (Qwen2-style) */
    const float *BQ = (const float *)QWEN2_0_5B_DECODE_PTR(model, L->bq);
    const float *BK = (const float *)QWEN2_0_5B_DECODE_PTR(model, L->bk);
    const float *BV = (const float *)QWEN2_0_5B_DECODE_PTR(model, L->bv);

    float *rope_cos = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_GLOBALS.rope_cos_cache);
    float *rope_sin = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_GLOBALS.rope_sin_cache);

    const int H = QWEN2_0_5B_DECODE_NUM_HEADS;
    const int H_kv = QWEN2_0_5B_DECODE_NUM_KV_HEADS;
    const int head_dim = QWEN2_0_5B_DECODE_HEAD_DIM;

    float q_token[H * aligned_head_dim];
    float k_token[H_kv * aligned_head_dim];
    float v_token[H_kv * aligned_head_dim];
    float attn_token[H * aligned_head_dim];

    /* Local MLP buffers (avoid layout dependencies for intermediate values) */
    float fc1_out[2 * aligned_intermediate_dim];
    float swiglu_out[aligned_intermediate_dim];

    /* Step 1: RMSNorm before attention */
    rmsnorm_forward(input,
                    ln1_gamma,
                    ln1_out,
                    NULL,
                    1,
                    QWEN2_0_5B_DECODE_EMBED_DIM,
                    aligned_embed_dim,
                    1e-06f);

    /* Step 2: QKV projection */
    /* Q projection: Q5_0 -> gemm_nt_q5_0 */
    gemm_nt_q5_0(ln1_out, WQ, BQ, q_token, 1, H * head_dim, aligned_embed_dim);

    /* K projection: Q5_0 -> gemm_nt_q5_0 */
    gemm_nt_q5_0(ln1_out, WK, BK, k_token, 1, H_kv * head_dim, aligned_embed_dim);

    /* V projection: Q5_0 -> gemm_nt_q5_0 */
    gemm_nt_q5_0(ln1_out, WV, BV, v_token, 1, H_kv * head_dim, aligned_embed_dim);

    /* Step 3: RoPE */
    rope_forward_qk(q_token,
                    k_token,
                    rope_cos,
                    rope_sin,
                    H,
                    H_kv,
                    1,
                    head_dim,
                    aligned_head_dim,
                    token_index);

    /* Step 4: KV cache write */
    kv_cache_write_head_major(k_token,
                              v_token,
                              k_cache,
                              v_cache,
                              H_kv,
                              token_index,
                              aligned_context_window,
                              head_dim,
                              aligned_head_dim);

    /* Step 5: Attention (decode) */
    attention_forward_decode_head_major_gqa_regular(q_token,
                                                   k_cache,
                                                   v_cache,
                                                   attn_token,
                                                   H,
                                                   H_kv,
                                                   token_index + 1,
                                                   aligned_context_window,
                                                   head_dim,
                                                   aligned_head_dim);

    /* Step 6: Output projection */
    /* WO projection: Q5_0 -> gemm_nt_q5_0 */
    gemm_nt_q5_0(attn_token, WO, NULL, proj_tmp, 1, aligned_embed_dim, H * head_dim);

    /* Step 7: Residual add */
    qwen2_0_5b_decode_residual_add_token_major(input, proj_tmp, residual1, 1, aligned_embed_dim);

    /* Step 8: RMSNorm before MLP */
    rmsnorm_forward(residual1,
                    ln2_gamma,
                    ln2_out,
                    NULL,
                    1,
                    QWEN2_0_5B_DECODE_EMBED_DIM,
                    aligned_embed_dim,
                    1e-06f);

    /* Step 9: MLP (SwiGLU) */
    /* Gate+Up projection: Q5_0 -> gemm_nt_q5_0 */
    gemm_nt_q5_0(ln2_out, W1, NULL, fc1_out, 1, 2 * aligned_intermediate_dim, aligned_embed_dim);

    /* SwiGLU activation */
    swiglu_forward(fc1_out, swiglu_out, 1, aligned_intermediate_dim);

    /* Down projection: Q4_K -> gemm_nt_q4_k */
    gemm_nt_q4_k(swiglu_out, W2, NULL, mlp_out, 1, aligned_embed_dim, aligned_intermediate_dim);

    /* Step 10: Final residual add */
    qwen2_0_5b_decode_residual_add_token_major(residual1, mlp_out, output, 1, aligned_embed_dim);
}

/*
 * Layer 3: wq=q5_0 wk=q5_0 wv=q8_0 wo=q5_0 w1=q5_0 w2=q6_k
 */
static void qwen2_0_5b_decode_layer_3_decode(
    QWEN2_0_5B_DECODEModel *model,
    int token_index,
    int aligned_embed_dim,
    int aligned_head_dim,
    int aligned_intermediate_dim,
    int aligned_context_window
) {
    const QWEN2_0_5B_DECODELayerOffsets *L = &QWEN2_0_5B_DECODE_LAYERS[3];

    float *input = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_LAYERS[2].output);

    float *ln1_gamma = QWEN2_0_5B_DECODE_PTR(model, L->ln1_gamma);
    float *ln1_out = QWEN2_0_5B_DECODE_PTR(model, L->ln1_out);
    float *ln2_gamma = QWEN2_0_5B_DECODE_PTR(model, L->ln2_gamma);
    float *ln2_out = QWEN2_0_5B_DECODE_PTR(model, L->ln2_out);
    float *k_cache = QWEN2_0_5B_DECODE_PTR(model, L->k);
    float *v_cache = QWEN2_0_5B_DECODE_PTR(model, L->v);
    float *proj_tmp = QWEN2_0_5B_DECODE_PTR(model, L->proj_tmp);
    float *proj_scratch = QWEN2_0_5B_DECODE_PTR(model, L->proj_scratch);
    float *residual1 = QWEN2_0_5B_DECODE_PTR(model, L->residual1);
    float *mlp_out = QWEN2_0_5B_DECODE_PTR(model, L->mlp_out);
    float *output = QWEN2_0_5B_DECODE_PTR(model, L->output);

    /* Weights (explicit types for layer 3) */
    const void *WQ = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wq);  /* Q5_0 */
    const void *WK = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wk);  /* Q5_0 */
    const void *WV = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wv);  /* Q8_0 */
    const void *WO = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wo);  /* Q5_0 */
    const void *W1 = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->w1);  /* Q5_0 (gate+up) */
    const void *W2 = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->w2);  /* Q6_K (down) */

    /* Attention biases (Qwen2-style) */
    const float *BQ = (const float *)QWEN2_0_5B_DECODE_PTR(model, L->bq);
    const float *BK = (const float *)QWEN2_0_5B_DECODE_PTR(model, L->bk);
    const float *BV = (const float *)QWEN2_0_5B_DECODE_PTR(model, L->bv);

    float *rope_cos = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_GLOBALS.rope_cos_cache);
    float *rope_sin = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_GLOBALS.rope_sin_cache);

    const int H = QWEN2_0_5B_DECODE_NUM_HEADS;
    const int H_kv = QWEN2_0_5B_DECODE_NUM_KV_HEADS;
    const int head_dim = QWEN2_0_5B_DECODE_HEAD_DIM;

    float q_token[H * aligned_head_dim];
    float k_token[H_kv * aligned_head_dim];
    float v_token[H_kv * aligned_head_dim];
    float attn_token[H * aligned_head_dim];

    /* Local MLP buffers (avoid layout dependencies for intermediate values) */
    float fc1_out[2 * aligned_intermediate_dim];
    float swiglu_out[aligned_intermediate_dim];

    /* Step 1: RMSNorm before attention */
    rmsnorm_forward(input,
                    ln1_gamma,
                    ln1_out,
                    NULL,
                    1,
                    QWEN2_0_5B_DECODE_EMBED_DIM,
                    aligned_embed_dim,
                    1e-06f);

    /* Step 2: QKV projection */
    /* Q projection: Q5_0 -> gemm_nt_q5_0 */
    gemm_nt_q5_0(ln1_out, WQ, BQ, q_token, 1, H * head_dim, aligned_embed_dim);

    /* K projection: Q5_0 -> gemm_nt_q5_0 */
    gemm_nt_q5_0(ln1_out, WK, BK, k_token, 1, H_kv * head_dim, aligned_embed_dim);

    /* V projection: Q8_0 -> gemm_nt_q8_0 */
    gemm_nt_q8_0(ln1_out, WV, BV, v_token, 1, H_kv * head_dim, aligned_embed_dim);

    /* Step 3: RoPE */
    rope_forward_qk(q_token,
                    k_token,
                    rope_cos,
                    rope_sin,
                    H,
                    H_kv,
                    1,
                    head_dim,
                    aligned_head_dim,
                    token_index);

    /* Step 4: KV cache write */
    kv_cache_write_head_major(k_token,
                              v_token,
                              k_cache,
                              v_cache,
                              H_kv,
                              token_index,
                              aligned_context_window,
                              head_dim,
                              aligned_head_dim);

    /* Step 5: Attention (decode) */
    attention_forward_decode_head_major_gqa_regular(q_token,
                                                   k_cache,
                                                   v_cache,
                                                   attn_token,
                                                   H,
                                                   H_kv,
                                                   token_index + 1,
                                                   aligned_context_window,
                                                   head_dim,
                                                   aligned_head_dim);

    /* Step 6: Output projection */
    /* WO projection: Q5_0 -> gemm_nt_q5_0 */
    gemm_nt_q5_0(attn_token, WO, NULL, proj_tmp, 1, aligned_embed_dim, H * head_dim);

    /* Step 7: Residual add */
    qwen2_0_5b_decode_residual_add_token_major(input, proj_tmp, residual1, 1, aligned_embed_dim);

    /* Step 8: RMSNorm before MLP */
    rmsnorm_forward(residual1,
                    ln2_gamma,
                    ln2_out,
                    NULL,
                    1,
                    QWEN2_0_5B_DECODE_EMBED_DIM,
                    aligned_embed_dim,
                    1e-06f);

    /* Step 9: MLP (SwiGLU) */
    /* Gate+Up projection: Q5_0 -> gemm_nt_q5_0 */
    gemm_nt_q5_0(ln2_out, W1, NULL, fc1_out, 1, 2 * aligned_intermediate_dim, aligned_embed_dim);

    /* SwiGLU activation */
    swiglu_forward(fc1_out, swiglu_out, 1, aligned_intermediate_dim);

    /* Down projection: Q6_K -> gemm_nt_q6_k */
    gemm_nt_q6_k(swiglu_out, W2, NULL, mlp_out, 1, aligned_embed_dim, aligned_intermediate_dim);

    /* Step 10: Final residual add */
    qwen2_0_5b_decode_residual_add_token_major(residual1, mlp_out, output, 1, aligned_embed_dim);
}

/*
 * Layer 4: wq=q5_0 wk=q5_0 wv=q5_0 wo=q5_0 w1=q5_0 w2=q4_k
 */
static void qwen2_0_5b_decode_layer_4_decode(
    QWEN2_0_5B_DECODEModel *model,
    int token_index,
    int aligned_embed_dim,
    int aligned_head_dim,
    int aligned_intermediate_dim,
    int aligned_context_window
) {
    const QWEN2_0_5B_DECODELayerOffsets *L = &QWEN2_0_5B_DECODE_LAYERS[4];

    float *input = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_LAYERS[3].output);

    float *ln1_gamma = QWEN2_0_5B_DECODE_PTR(model, L->ln1_gamma);
    float *ln1_out = QWEN2_0_5B_DECODE_PTR(model, L->ln1_out);
    float *ln2_gamma = QWEN2_0_5B_DECODE_PTR(model, L->ln2_gamma);
    float *ln2_out = QWEN2_0_5B_DECODE_PTR(model, L->ln2_out);
    float *k_cache = QWEN2_0_5B_DECODE_PTR(model, L->k);
    float *v_cache = QWEN2_0_5B_DECODE_PTR(model, L->v);
    float *proj_tmp = QWEN2_0_5B_DECODE_PTR(model, L->proj_tmp);
    float *proj_scratch = QWEN2_0_5B_DECODE_PTR(model, L->proj_scratch);
    float *residual1 = QWEN2_0_5B_DECODE_PTR(model, L->residual1);
    float *mlp_out = QWEN2_0_5B_DECODE_PTR(model, L->mlp_out);
    float *output = QWEN2_0_5B_DECODE_PTR(model, L->output);

    /* Weights (explicit types for layer 4) */
    const void *WQ = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wq);  /* Q5_0 */
    const void *WK = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wk);  /* Q5_0 */
    const void *WV = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wv);  /* Q5_0 */
    const void *WO = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wo);  /* Q5_0 */
    const void *W1 = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->w1);  /* Q5_0 (gate+up) */
    const void *W2 = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->w2);  /* Q4_K (down) */

    /* Attention biases (Qwen2-style) */
    const float *BQ = (const float *)QWEN2_0_5B_DECODE_PTR(model, L->bq);
    const float *BK = (const float *)QWEN2_0_5B_DECODE_PTR(model, L->bk);
    const float *BV = (const float *)QWEN2_0_5B_DECODE_PTR(model, L->bv);

    float *rope_cos = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_GLOBALS.rope_cos_cache);
    float *rope_sin = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_GLOBALS.rope_sin_cache);

    const int H = QWEN2_0_5B_DECODE_NUM_HEADS;
    const int H_kv = QWEN2_0_5B_DECODE_NUM_KV_HEADS;
    const int head_dim = QWEN2_0_5B_DECODE_HEAD_DIM;

    float q_token[H * aligned_head_dim];
    float k_token[H_kv * aligned_head_dim];
    float v_token[H_kv * aligned_head_dim];
    float attn_token[H * aligned_head_dim];

    /* Local MLP buffers (avoid layout dependencies for intermediate values) */
    float fc1_out[2 * aligned_intermediate_dim];
    float swiglu_out[aligned_intermediate_dim];

    /* Step 1: RMSNorm before attention */
    rmsnorm_forward(input,
                    ln1_gamma,
                    ln1_out,
                    NULL,
                    1,
                    QWEN2_0_5B_DECODE_EMBED_DIM,
                    aligned_embed_dim,
                    1e-06f);

    /* Step 2: QKV projection */
    /* Q projection: Q5_0 -> gemm_nt_q5_0 */
    gemm_nt_q5_0(ln1_out, WQ, BQ, q_token, 1, H * head_dim, aligned_embed_dim);

    /* K projection: Q5_0 -> gemm_nt_q5_0 */
    gemm_nt_q5_0(ln1_out, WK, BK, k_token, 1, H_kv * head_dim, aligned_embed_dim);

    /* V projection: Q5_0 -> gemm_nt_q5_0 */
    gemm_nt_q5_0(ln1_out, WV, BV, v_token, 1, H_kv * head_dim, aligned_embed_dim);

    /* Step 3: RoPE */
    rope_forward_qk(q_token,
                    k_token,
                    rope_cos,
                    rope_sin,
                    H,
                    H_kv,
                    1,
                    head_dim,
                    aligned_head_dim,
                    token_index);

    /* Step 4: KV cache write */
    kv_cache_write_head_major(k_token,
                              v_token,
                              k_cache,
                              v_cache,
                              H_kv,
                              token_index,
                              aligned_context_window,
                              head_dim,
                              aligned_head_dim);

    /* Step 5: Attention (decode) */
    attention_forward_decode_head_major_gqa_regular(q_token,
                                                   k_cache,
                                                   v_cache,
                                                   attn_token,
                                                   H,
                                                   H_kv,
                                                   token_index + 1,
                                                   aligned_context_window,
                                                   head_dim,
                                                   aligned_head_dim);

    /* Step 6: Output projection */
    /* WO projection: Q5_0 -> gemm_nt_q5_0 */
    gemm_nt_q5_0(attn_token, WO, NULL, proj_tmp, 1, aligned_embed_dim, H * head_dim);

    /* Step 7: Residual add */
    qwen2_0_5b_decode_residual_add_token_major(input, proj_tmp, residual1, 1, aligned_embed_dim);

    /* Step 8: RMSNorm before MLP */
    rmsnorm_forward(residual1,
                    ln2_gamma,
                    ln2_out,
                    NULL,
                    1,
                    QWEN2_0_5B_DECODE_EMBED_DIM,
                    aligned_embed_dim,
                    1e-06f);

    /* Step 9: MLP (SwiGLU) */
    /* Gate+Up projection: Q5_0 -> gemm_nt_q5_0 */
    gemm_nt_q5_0(ln2_out, W1, NULL, fc1_out, 1, 2 * aligned_intermediate_dim, aligned_embed_dim);

    /* SwiGLU activation */
    swiglu_forward(fc1_out, swiglu_out, 1, aligned_intermediate_dim);

    /* Down projection: Q4_K -> gemm_nt_q4_k */
    gemm_nt_q4_k(swiglu_out, W2, NULL, mlp_out, 1, aligned_embed_dim, aligned_intermediate_dim);

    /* Step 10: Final residual add */
    qwen2_0_5b_decode_residual_add_token_major(residual1, mlp_out, output, 1, aligned_embed_dim);
}

/*
 * Layer 5: wq=q5_0 wk=q5_0 wv=q5_0 wo=q5_0 w1=q5_0 w2=q4_k
 */
static void qwen2_0_5b_decode_layer_5_decode(
    QWEN2_0_5B_DECODEModel *model,
    int token_index,
    int aligned_embed_dim,
    int aligned_head_dim,
    int aligned_intermediate_dim,
    int aligned_context_window
) {
    const QWEN2_0_5B_DECODELayerOffsets *L = &QWEN2_0_5B_DECODE_LAYERS[5];

    float *input = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_LAYERS[4].output);

    float *ln1_gamma = QWEN2_0_5B_DECODE_PTR(model, L->ln1_gamma);
    float *ln1_out = QWEN2_0_5B_DECODE_PTR(model, L->ln1_out);
    float *ln2_gamma = QWEN2_0_5B_DECODE_PTR(model, L->ln2_gamma);
    float *ln2_out = QWEN2_0_5B_DECODE_PTR(model, L->ln2_out);
    float *k_cache = QWEN2_0_5B_DECODE_PTR(model, L->k);
    float *v_cache = QWEN2_0_5B_DECODE_PTR(model, L->v);
    float *proj_tmp = QWEN2_0_5B_DECODE_PTR(model, L->proj_tmp);
    float *proj_scratch = QWEN2_0_5B_DECODE_PTR(model, L->proj_scratch);
    float *residual1 = QWEN2_0_5B_DECODE_PTR(model, L->residual1);
    float *mlp_out = QWEN2_0_5B_DECODE_PTR(model, L->mlp_out);
    float *output = QWEN2_0_5B_DECODE_PTR(model, L->output);

    /* Weights (explicit types for layer 5) */
    const void *WQ = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wq);  /* Q5_0 */
    const void *WK = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wk);  /* Q5_0 */
    const void *WV = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wv);  /* Q5_0 */
    const void *WO = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wo);  /* Q5_0 */
    const void *W1 = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->w1);  /* Q5_0 (gate+up) */
    const void *W2 = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->w2);  /* Q4_K (down) */

    /* Attention biases (Qwen2-style) */
    const float *BQ = (const float *)QWEN2_0_5B_DECODE_PTR(model, L->bq);
    const float *BK = (const float *)QWEN2_0_5B_DECODE_PTR(model, L->bk);
    const float *BV = (const float *)QWEN2_0_5B_DECODE_PTR(model, L->bv);

    float *rope_cos = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_GLOBALS.rope_cos_cache);
    float *rope_sin = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_GLOBALS.rope_sin_cache);

    const int H = QWEN2_0_5B_DECODE_NUM_HEADS;
    const int H_kv = QWEN2_0_5B_DECODE_NUM_KV_HEADS;
    const int head_dim = QWEN2_0_5B_DECODE_HEAD_DIM;

    float q_token[H * aligned_head_dim];
    float k_token[H_kv * aligned_head_dim];
    float v_token[H_kv * aligned_head_dim];
    float attn_token[H * aligned_head_dim];

    /* Local MLP buffers (avoid layout dependencies for intermediate values) */
    float fc1_out[2 * aligned_intermediate_dim];
    float swiglu_out[aligned_intermediate_dim];

    /* Step 1: RMSNorm before attention */
    rmsnorm_forward(input,
                    ln1_gamma,
                    ln1_out,
                    NULL,
                    1,
                    QWEN2_0_5B_DECODE_EMBED_DIM,
                    aligned_embed_dim,
                    1e-06f);

    /* Step 2: QKV projection */
    /* Q projection: Q5_0 -> gemm_nt_q5_0 */
    gemm_nt_q5_0(ln1_out, WQ, BQ, q_token, 1, H * head_dim, aligned_embed_dim);

    /* K projection: Q5_0 -> gemm_nt_q5_0 */
    gemm_nt_q5_0(ln1_out, WK, BK, k_token, 1, H_kv * head_dim, aligned_embed_dim);

    /* V projection: Q5_0 -> gemm_nt_q5_0 */
    gemm_nt_q5_0(ln1_out, WV, BV, v_token, 1, H_kv * head_dim, aligned_embed_dim);

    /* Step 3: RoPE */
    rope_forward_qk(q_token,
                    k_token,
                    rope_cos,
                    rope_sin,
                    H,
                    H_kv,
                    1,
                    head_dim,
                    aligned_head_dim,
                    token_index);

    /* Step 4: KV cache write */
    kv_cache_write_head_major(k_token,
                              v_token,
                              k_cache,
                              v_cache,
                              H_kv,
                              token_index,
                              aligned_context_window,
                              head_dim,
                              aligned_head_dim);

    /* Step 5: Attention (decode) */
    attention_forward_decode_head_major_gqa_regular(q_token,
                                                   k_cache,
                                                   v_cache,
                                                   attn_token,
                                                   H,
                                                   H_kv,
                                                   token_index + 1,
                                                   aligned_context_window,
                                                   head_dim,
                                                   aligned_head_dim);

    /* Step 6: Output projection */
    /* WO projection: Q5_0 -> gemm_nt_q5_0 */
    gemm_nt_q5_0(attn_token, WO, NULL, proj_tmp, 1, aligned_embed_dim, H * head_dim);

    /* Step 7: Residual add */
    qwen2_0_5b_decode_residual_add_token_major(input, proj_tmp, residual1, 1, aligned_embed_dim);

    /* Step 8: RMSNorm before MLP */
    rmsnorm_forward(residual1,
                    ln2_gamma,
                    ln2_out,
                    NULL,
                    1,
                    QWEN2_0_5B_DECODE_EMBED_DIM,
                    aligned_embed_dim,
                    1e-06f);

    /* Step 9: MLP (SwiGLU) */
    /* Gate+Up projection: Q5_0 -> gemm_nt_q5_0 */
    gemm_nt_q5_0(ln2_out, W1, NULL, fc1_out, 1, 2 * aligned_intermediate_dim, aligned_embed_dim);

    /* SwiGLU activation */
    swiglu_forward(fc1_out, swiglu_out, 1, aligned_intermediate_dim);

    /* Down projection: Q4_K -> gemm_nt_q4_k */
    gemm_nt_q4_k(swiglu_out, W2, NULL, mlp_out, 1, aligned_embed_dim, aligned_intermediate_dim);

    /* Step 10: Final residual add */
    qwen2_0_5b_decode_residual_add_token_major(residual1, mlp_out, output, 1, aligned_embed_dim);
}

/*
 * Layer 6: wq=q5_0 wk=q5_0 wv=q8_0 wo=q5_0 w1=q5_0 w2=q6_k
 */
static void qwen2_0_5b_decode_layer_6_decode(
    QWEN2_0_5B_DECODEModel *model,
    int token_index,
    int aligned_embed_dim,
    int aligned_head_dim,
    int aligned_intermediate_dim,
    int aligned_context_window
) {
    const QWEN2_0_5B_DECODELayerOffsets *L = &QWEN2_0_5B_DECODE_LAYERS[6];

    float *input = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_LAYERS[5].output);

    float *ln1_gamma = QWEN2_0_5B_DECODE_PTR(model, L->ln1_gamma);
    float *ln1_out = QWEN2_0_5B_DECODE_PTR(model, L->ln1_out);
    float *ln2_gamma = QWEN2_0_5B_DECODE_PTR(model, L->ln2_gamma);
    float *ln2_out = QWEN2_0_5B_DECODE_PTR(model, L->ln2_out);
    float *k_cache = QWEN2_0_5B_DECODE_PTR(model, L->k);
    float *v_cache = QWEN2_0_5B_DECODE_PTR(model, L->v);
    float *proj_tmp = QWEN2_0_5B_DECODE_PTR(model, L->proj_tmp);
    float *proj_scratch = QWEN2_0_5B_DECODE_PTR(model, L->proj_scratch);
    float *residual1 = QWEN2_0_5B_DECODE_PTR(model, L->residual1);
    float *mlp_out = QWEN2_0_5B_DECODE_PTR(model, L->mlp_out);
    float *output = QWEN2_0_5B_DECODE_PTR(model, L->output);

    /* Weights (explicit types for layer 6) */
    const void *WQ = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wq);  /* Q5_0 */
    const void *WK = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wk);  /* Q5_0 */
    const void *WV = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wv);  /* Q8_0 */
    const void *WO = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wo);  /* Q5_0 */
    const void *W1 = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->w1);  /* Q5_0 (gate+up) */
    const void *W2 = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->w2);  /* Q6_K (down) */

    /* Attention biases (Qwen2-style) */
    const float *BQ = (const float *)QWEN2_0_5B_DECODE_PTR(model, L->bq);
    const float *BK = (const float *)QWEN2_0_5B_DECODE_PTR(model, L->bk);
    const float *BV = (const float *)QWEN2_0_5B_DECODE_PTR(model, L->bv);

    float *rope_cos = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_GLOBALS.rope_cos_cache);
    float *rope_sin = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_GLOBALS.rope_sin_cache);

    const int H = QWEN2_0_5B_DECODE_NUM_HEADS;
    const int H_kv = QWEN2_0_5B_DECODE_NUM_KV_HEADS;
    const int head_dim = QWEN2_0_5B_DECODE_HEAD_DIM;

    float q_token[H * aligned_head_dim];
    float k_token[H_kv * aligned_head_dim];
    float v_token[H_kv * aligned_head_dim];
    float attn_token[H * aligned_head_dim];

    /* Local MLP buffers (avoid layout dependencies for intermediate values) */
    float fc1_out[2 * aligned_intermediate_dim];
    float swiglu_out[aligned_intermediate_dim];

    /* Step 1: RMSNorm before attention */
    rmsnorm_forward(input,
                    ln1_gamma,
                    ln1_out,
                    NULL,
                    1,
                    QWEN2_0_5B_DECODE_EMBED_DIM,
                    aligned_embed_dim,
                    1e-06f);

    /* Step 2: QKV projection */
    /* Q projection: Q5_0 -> gemm_nt_q5_0 */
    gemm_nt_q5_0(ln1_out, WQ, BQ, q_token, 1, H * head_dim, aligned_embed_dim);

    /* K projection: Q5_0 -> gemm_nt_q5_0 */
    gemm_nt_q5_0(ln1_out, WK, BK, k_token, 1, H_kv * head_dim, aligned_embed_dim);

    /* V projection: Q8_0 -> gemm_nt_q8_0 */
    gemm_nt_q8_0(ln1_out, WV, BV, v_token, 1, H_kv * head_dim, aligned_embed_dim);

    /* Step 3: RoPE */
    rope_forward_qk(q_token,
                    k_token,
                    rope_cos,
                    rope_sin,
                    H,
                    H_kv,
                    1,
                    head_dim,
                    aligned_head_dim,
                    token_index);

    /* Step 4: KV cache write */
    kv_cache_write_head_major(k_token,
                              v_token,
                              k_cache,
                              v_cache,
                              H_kv,
                              token_index,
                              aligned_context_window,
                              head_dim,
                              aligned_head_dim);

    /* Step 5: Attention (decode) */
    attention_forward_decode_head_major_gqa_regular(q_token,
                                                   k_cache,
                                                   v_cache,
                                                   attn_token,
                                                   H,
                                                   H_kv,
                                                   token_index + 1,
                                                   aligned_context_window,
                                                   head_dim,
                                                   aligned_head_dim);

    /* Step 6: Output projection */
    /* WO projection: Q5_0 -> gemm_nt_q5_0 */
    gemm_nt_q5_0(attn_token, WO, NULL, proj_tmp, 1, aligned_embed_dim, H * head_dim);

    /* Step 7: Residual add */
    qwen2_0_5b_decode_residual_add_token_major(input, proj_tmp, residual1, 1, aligned_embed_dim);

    /* Step 8: RMSNorm before MLP */
    rmsnorm_forward(residual1,
                    ln2_gamma,
                    ln2_out,
                    NULL,
                    1,
                    QWEN2_0_5B_DECODE_EMBED_DIM,
                    aligned_embed_dim,
                    1e-06f);

    /* Step 9: MLP (SwiGLU) */
    /* Gate+Up projection: Q5_0 -> gemm_nt_q5_0 */
    gemm_nt_q5_0(ln2_out, W1, NULL, fc1_out, 1, 2 * aligned_intermediate_dim, aligned_embed_dim);

    /* SwiGLU activation */
    swiglu_forward(fc1_out, swiglu_out, 1, aligned_intermediate_dim);

    /* Down projection: Q6_K -> gemm_nt_q6_k */
    gemm_nt_q6_k(swiglu_out, W2, NULL, mlp_out, 1, aligned_embed_dim, aligned_intermediate_dim);

    /* Step 10: Final residual add */
    qwen2_0_5b_decode_residual_add_token_major(residual1, mlp_out, output, 1, aligned_embed_dim);
}

/*
 * Layer 7: wq=q5_0 wk=q5_0 wv=q8_0 wo=q5_0 w1=q5_0 w2=q6_k
 */
static void qwen2_0_5b_decode_layer_7_decode(
    QWEN2_0_5B_DECODEModel *model,
    int token_index,
    int aligned_embed_dim,
    int aligned_head_dim,
    int aligned_intermediate_dim,
    int aligned_context_window
) {
    const QWEN2_0_5B_DECODELayerOffsets *L = &QWEN2_0_5B_DECODE_LAYERS[7];

    float *input = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_LAYERS[6].output);

    float *ln1_gamma = QWEN2_0_5B_DECODE_PTR(model, L->ln1_gamma);
    float *ln1_out = QWEN2_0_5B_DECODE_PTR(model, L->ln1_out);
    float *ln2_gamma = QWEN2_0_5B_DECODE_PTR(model, L->ln2_gamma);
    float *ln2_out = QWEN2_0_5B_DECODE_PTR(model, L->ln2_out);
    float *k_cache = QWEN2_0_5B_DECODE_PTR(model, L->k);
    float *v_cache = QWEN2_0_5B_DECODE_PTR(model, L->v);
    float *proj_tmp = QWEN2_0_5B_DECODE_PTR(model, L->proj_tmp);
    float *proj_scratch = QWEN2_0_5B_DECODE_PTR(model, L->proj_scratch);
    float *residual1 = QWEN2_0_5B_DECODE_PTR(model, L->residual1);
    float *mlp_out = QWEN2_0_5B_DECODE_PTR(model, L->mlp_out);
    float *output = QWEN2_0_5B_DECODE_PTR(model, L->output);

    /* Weights (explicit types for layer 7) */
    const void *WQ = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wq);  /* Q5_0 */
    const void *WK = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wk);  /* Q5_0 */
    const void *WV = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wv);  /* Q8_0 */
    const void *WO = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wo);  /* Q5_0 */
    const void *W1 = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->w1);  /* Q5_0 (gate+up) */
    const void *W2 = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->w2);  /* Q6_K (down) */

    /* Attention biases (Qwen2-style) */
    const float *BQ = (const float *)QWEN2_0_5B_DECODE_PTR(model, L->bq);
    const float *BK = (const float *)QWEN2_0_5B_DECODE_PTR(model, L->bk);
    const float *BV = (const float *)QWEN2_0_5B_DECODE_PTR(model, L->bv);

    float *rope_cos = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_GLOBALS.rope_cos_cache);
    float *rope_sin = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_GLOBALS.rope_sin_cache);

    const int H = QWEN2_0_5B_DECODE_NUM_HEADS;
    const int H_kv = QWEN2_0_5B_DECODE_NUM_KV_HEADS;
    const int head_dim = QWEN2_0_5B_DECODE_HEAD_DIM;

    float q_token[H * aligned_head_dim];
    float k_token[H_kv * aligned_head_dim];
    float v_token[H_kv * aligned_head_dim];
    float attn_token[H * aligned_head_dim];

    /* Local MLP buffers (avoid layout dependencies for intermediate values) */
    float fc1_out[2 * aligned_intermediate_dim];
    float swiglu_out[aligned_intermediate_dim];

    /* Step 1: RMSNorm before attention */
    rmsnorm_forward(input,
                    ln1_gamma,
                    ln1_out,
                    NULL,
                    1,
                    QWEN2_0_5B_DECODE_EMBED_DIM,
                    aligned_embed_dim,
                    1e-06f);

    /* Step 2: QKV projection */
    /* Q projection: Q5_0 -> gemm_nt_q5_0 */
    gemm_nt_q5_0(ln1_out, WQ, BQ, q_token, 1, H * head_dim, aligned_embed_dim);

    /* K projection: Q5_0 -> gemm_nt_q5_0 */
    gemm_nt_q5_0(ln1_out, WK, BK, k_token, 1, H_kv * head_dim, aligned_embed_dim);

    /* V projection: Q8_0 -> gemm_nt_q8_0 */
    gemm_nt_q8_0(ln1_out, WV, BV, v_token, 1, H_kv * head_dim, aligned_embed_dim);

    /* Step 3: RoPE */
    rope_forward_qk(q_token,
                    k_token,
                    rope_cos,
                    rope_sin,
                    H,
                    H_kv,
                    1,
                    head_dim,
                    aligned_head_dim,
                    token_index);

    /* Step 4: KV cache write */
    kv_cache_write_head_major(k_token,
                              v_token,
                              k_cache,
                              v_cache,
                              H_kv,
                              token_index,
                              aligned_context_window,
                              head_dim,
                              aligned_head_dim);

    /* Step 5: Attention (decode) */
    attention_forward_decode_head_major_gqa_regular(q_token,
                                                   k_cache,
                                                   v_cache,
                                                   attn_token,
                                                   H,
                                                   H_kv,
                                                   token_index + 1,
                                                   aligned_context_window,
                                                   head_dim,
                                                   aligned_head_dim);

    /* Step 6: Output projection */
    /* WO projection: Q5_0 -> gemm_nt_q5_0 */
    gemm_nt_q5_0(attn_token, WO, NULL, proj_tmp, 1, aligned_embed_dim, H * head_dim);

    /* Step 7: Residual add */
    qwen2_0_5b_decode_residual_add_token_major(input, proj_tmp, residual1, 1, aligned_embed_dim);

    /* Step 8: RMSNorm before MLP */
    rmsnorm_forward(residual1,
                    ln2_gamma,
                    ln2_out,
                    NULL,
                    1,
                    QWEN2_0_5B_DECODE_EMBED_DIM,
                    aligned_embed_dim,
                    1e-06f);

    /* Step 9: MLP (SwiGLU) */
    /* Gate+Up projection: Q5_0 -> gemm_nt_q5_0 */
    gemm_nt_q5_0(ln2_out, W1, NULL, fc1_out, 1, 2 * aligned_intermediate_dim, aligned_embed_dim);

    /* SwiGLU activation */
    swiglu_forward(fc1_out, swiglu_out, 1, aligned_intermediate_dim);

    /* Down projection: Q6_K -> gemm_nt_q6_k */
    gemm_nt_q6_k(swiglu_out, W2, NULL, mlp_out, 1, aligned_embed_dim, aligned_intermediate_dim);

    /* Step 10: Final residual add */
    qwen2_0_5b_decode_residual_add_token_major(residual1, mlp_out, output, 1, aligned_embed_dim);
}

/*
 * Layer 8: wq=q5_0 wk=q5_0 wv=q8_0 wo=q5_0 w1=q5_0 w2=q6_k
 */
static void qwen2_0_5b_decode_layer_8_decode(
    QWEN2_0_5B_DECODEModel *model,
    int token_index,
    int aligned_embed_dim,
    int aligned_head_dim,
    int aligned_intermediate_dim,
    int aligned_context_window
) {
    const QWEN2_0_5B_DECODELayerOffsets *L = &QWEN2_0_5B_DECODE_LAYERS[8];

    float *input = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_LAYERS[7].output);

    float *ln1_gamma = QWEN2_0_5B_DECODE_PTR(model, L->ln1_gamma);
    float *ln1_out = QWEN2_0_5B_DECODE_PTR(model, L->ln1_out);
    float *ln2_gamma = QWEN2_0_5B_DECODE_PTR(model, L->ln2_gamma);
    float *ln2_out = QWEN2_0_5B_DECODE_PTR(model, L->ln2_out);
    float *k_cache = QWEN2_0_5B_DECODE_PTR(model, L->k);
    float *v_cache = QWEN2_0_5B_DECODE_PTR(model, L->v);
    float *proj_tmp = QWEN2_0_5B_DECODE_PTR(model, L->proj_tmp);
    float *proj_scratch = QWEN2_0_5B_DECODE_PTR(model, L->proj_scratch);
    float *residual1 = QWEN2_0_5B_DECODE_PTR(model, L->residual1);
    float *mlp_out = QWEN2_0_5B_DECODE_PTR(model, L->mlp_out);
    float *output = QWEN2_0_5B_DECODE_PTR(model, L->output);

    /* Weights (explicit types for layer 8) */
    const void *WQ = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wq);  /* Q5_0 */
    const void *WK = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wk);  /* Q5_0 */
    const void *WV = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wv);  /* Q8_0 */
    const void *WO = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wo);  /* Q5_0 */
    const void *W1 = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->w1);  /* Q5_0 (gate+up) */
    const void *W2 = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->w2);  /* Q6_K (down) */

    /* Attention biases (Qwen2-style) */
    const float *BQ = (const float *)QWEN2_0_5B_DECODE_PTR(model, L->bq);
    const float *BK = (const float *)QWEN2_0_5B_DECODE_PTR(model, L->bk);
    const float *BV = (const float *)QWEN2_0_5B_DECODE_PTR(model, L->bv);

    float *rope_cos = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_GLOBALS.rope_cos_cache);
    float *rope_sin = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_GLOBALS.rope_sin_cache);

    const int H = QWEN2_0_5B_DECODE_NUM_HEADS;
    const int H_kv = QWEN2_0_5B_DECODE_NUM_KV_HEADS;
    const int head_dim = QWEN2_0_5B_DECODE_HEAD_DIM;

    float q_token[H * aligned_head_dim];
    float k_token[H_kv * aligned_head_dim];
    float v_token[H_kv * aligned_head_dim];
    float attn_token[H * aligned_head_dim];

    /* Local MLP buffers (avoid layout dependencies for intermediate values) */
    float fc1_out[2 * aligned_intermediate_dim];
    float swiglu_out[aligned_intermediate_dim];

    /* Step 1: RMSNorm before attention */
    rmsnorm_forward(input,
                    ln1_gamma,
                    ln1_out,
                    NULL,
                    1,
                    QWEN2_0_5B_DECODE_EMBED_DIM,
                    aligned_embed_dim,
                    1e-06f);

    /* Step 2: QKV projection */
    /* Q projection: Q5_0 -> gemm_nt_q5_0 */
    gemm_nt_q5_0(ln1_out, WQ, BQ, q_token, 1, H * head_dim, aligned_embed_dim);

    /* K projection: Q5_0 -> gemm_nt_q5_0 */
    gemm_nt_q5_0(ln1_out, WK, BK, k_token, 1, H_kv * head_dim, aligned_embed_dim);

    /* V projection: Q8_0 -> gemm_nt_q8_0 */
    gemm_nt_q8_0(ln1_out, WV, BV, v_token, 1, H_kv * head_dim, aligned_embed_dim);

    /* Step 3: RoPE */
    rope_forward_qk(q_token,
                    k_token,
                    rope_cos,
                    rope_sin,
                    H,
                    H_kv,
                    1,
                    head_dim,
                    aligned_head_dim,
                    token_index);

    /* Step 4: KV cache write */
    kv_cache_write_head_major(k_token,
                              v_token,
                              k_cache,
                              v_cache,
                              H_kv,
                              token_index,
                              aligned_context_window,
                              head_dim,
                              aligned_head_dim);

    /* Step 5: Attention (decode) */
    attention_forward_decode_head_major_gqa_regular(q_token,
                                                   k_cache,
                                                   v_cache,
                                                   attn_token,
                                                   H,
                                                   H_kv,
                                                   token_index + 1,
                                                   aligned_context_window,
                                                   head_dim,
                                                   aligned_head_dim);

    /* Step 6: Output projection */
    /* WO projection: Q5_0 -> gemm_nt_q5_0 */
    gemm_nt_q5_0(attn_token, WO, NULL, proj_tmp, 1, aligned_embed_dim, H * head_dim);

    /* Step 7: Residual add */
    qwen2_0_5b_decode_residual_add_token_major(input, proj_tmp, residual1, 1, aligned_embed_dim);

    /* Step 8: RMSNorm before MLP */
    rmsnorm_forward(residual1,
                    ln2_gamma,
                    ln2_out,
                    NULL,
                    1,
                    QWEN2_0_5B_DECODE_EMBED_DIM,
                    aligned_embed_dim,
                    1e-06f);

    /* Step 9: MLP (SwiGLU) */
    /* Gate+Up projection: Q5_0 -> gemm_nt_q5_0 */
    gemm_nt_q5_0(ln2_out, W1, NULL, fc1_out, 1, 2 * aligned_intermediate_dim, aligned_embed_dim);

    /* SwiGLU activation */
    swiglu_forward(fc1_out, swiglu_out, 1, aligned_intermediate_dim);

    /* Down projection: Q6_K -> gemm_nt_q6_k */
    gemm_nt_q6_k(swiglu_out, W2, NULL, mlp_out, 1, aligned_embed_dim, aligned_intermediate_dim);

    /* Step 10: Final residual add */
    qwen2_0_5b_decode_residual_add_token_major(residual1, mlp_out, output, 1, aligned_embed_dim);
}

/*
 * Layer 9: wq=q5_0 wk=q5_0 wv=q8_0 wo=q5_0 w1=q5_0 w2=q6_k
 */
static void qwen2_0_5b_decode_layer_9_decode(
    QWEN2_0_5B_DECODEModel *model,
    int token_index,
    int aligned_embed_dim,
    int aligned_head_dim,
    int aligned_intermediate_dim,
    int aligned_context_window
) {
    const QWEN2_0_5B_DECODELayerOffsets *L = &QWEN2_0_5B_DECODE_LAYERS[9];

    float *input = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_LAYERS[8].output);

    float *ln1_gamma = QWEN2_0_5B_DECODE_PTR(model, L->ln1_gamma);
    float *ln1_out = QWEN2_0_5B_DECODE_PTR(model, L->ln1_out);
    float *ln2_gamma = QWEN2_0_5B_DECODE_PTR(model, L->ln2_gamma);
    float *ln2_out = QWEN2_0_5B_DECODE_PTR(model, L->ln2_out);
    float *k_cache = QWEN2_0_5B_DECODE_PTR(model, L->k);
    float *v_cache = QWEN2_0_5B_DECODE_PTR(model, L->v);
    float *proj_tmp = QWEN2_0_5B_DECODE_PTR(model, L->proj_tmp);
    float *proj_scratch = QWEN2_0_5B_DECODE_PTR(model, L->proj_scratch);
    float *residual1 = QWEN2_0_5B_DECODE_PTR(model, L->residual1);
    float *mlp_out = QWEN2_0_5B_DECODE_PTR(model, L->mlp_out);
    float *output = QWEN2_0_5B_DECODE_PTR(model, L->output);

    /* Weights (explicit types for layer 9) */
    const void *WQ = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wq);  /* Q5_0 */
    const void *WK = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wk);  /* Q5_0 */
    const void *WV = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wv);  /* Q8_0 */
    const void *WO = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wo);  /* Q5_0 */
    const void *W1 = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->w1);  /* Q5_0 (gate+up) */
    const void *W2 = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->w2);  /* Q6_K (down) */

    /* Attention biases (Qwen2-style) */
    const float *BQ = (const float *)QWEN2_0_5B_DECODE_PTR(model, L->bq);
    const float *BK = (const float *)QWEN2_0_5B_DECODE_PTR(model, L->bk);
    const float *BV = (const float *)QWEN2_0_5B_DECODE_PTR(model, L->bv);

    float *rope_cos = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_GLOBALS.rope_cos_cache);
    float *rope_sin = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_GLOBALS.rope_sin_cache);

    const int H = QWEN2_0_5B_DECODE_NUM_HEADS;
    const int H_kv = QWEN2_0_5B_DECODE_NUM_KV_HEADS;
    const int head_dim = QWEN2_0_5B_DECODE_HEAD_DIM;

    float q_token[H * aligned_head_dim];
    float k_token[H_kv * aligned_head_dim];
    float v_token[H_kv * aligned_head_dim];
    float attn_token[H * aligned_head_dim];

    /* Local MLP buffers (avoid layout dependencies for intermediate values) */
    float fc1_out[2 * aligned_intermediate_dim];
    float swiglu_out[aligned_intermediate_dim];

    /* Step 1: RMSNorm before attention */
    rmsnorm_forward(input,
                    ln1_gamma,
                    ln1_out,
                    NULL,
                    1,
                    QWEN2_0_5B_DECODE_EMBED_DIM,
                    aligned_embed_dim,
                    1e-06f);

    /* Step 2: QKV projection */
    /* Q projection: Q5_0 -> gemm_nt_q5_0 */
    gemm_nt_q5_0(ln1_out, WQ, BQ, q_token, 1, H * head_dim, aligned_embed_dim);

    /* K projection: Q5_0 -> gemm_nt_q5_0 */
    gemm_nt_q5_0(ln1_out, WK, BK, k_token, 1, H_kv * head_dim, aligned_embed_dim);

    /* V projection: Q8_0 -> gemm_nt_q8_0 */
    gemm_nt_q8_0(ln1_out, WV, BV, v_token, 1, H_kv * head_dim, aligned_embed_dim);

    /* Step 3: RoPE */
    rope_forward_qk(q_token,
                    k_token,
                    rope_cos,
                    rope_sin,
                    H,
                    H_kv,
                    1,
                    head_dim,
                    aligned_head_dim,
                    token_index);

    /* Step 4: KV cache write */
    kv_cache_write_head_major(k_token,
                              v_token,
                              k_cache,
                              v_cache,
                              H_kv,
                              token_index,
                              aligned_context_window,
                              head_dim,
                              aligned_head_dim);

    /* Step 5: Attention (decode) */
    attention_forward_decode_head_major_gqa_regular(q_token,
                                                   k_cache,
                                                   v_cache,
                                                   attn_token,
                                                   H,
                                                   H_kv,
                                                   token_index + 1,
                                                   aligned_context_window,
                                                   head_dim,
                                                   aligned_head_dim);

    /* Step 6: Output projection */
    /* WO projection: Q5_0 -> gemm_nt_q5_0 */
    gemm_nt_q5_0(attn_token, WO, NULL, proj_tmp, 1, aligned_embed_dim, H * head_dim);

    /* Step 7: Residual add */
    qwen2_0_5b_decode_residual_add_token_major(input, proj_tmp, residual1, 1, aligned_embed_dim);

    /* Step 8: RMSNorm before MLP */
    rmsnorm_forward(residual1,
                    ln2_gamma,
                    ln2_out,
                    NULL,
                    1,
                    QWEN2_0_5B_DECODE_EMBED_DIM,
                    aligned_embed_dim,
                    1e-06f);

    /* Step 9: MLP (SwiGLU) */
    /* Gate+Up projection: Q5_0 -> gemm_nt_q5_0 */
    gemm_nt_q5_0(ln2_out, W1, NULL, fc1_out, 1, 2 * aligned_intermediate_dim, aligned_embed_dim);

    /* SwiGLU activation */
    swiglu_forward(fc1_out, swiglu_out, 1, aligned_intermediate_dim);

    /* Down projection: Q6_K -> gemm_nt_q6_k */
    gemm_nt_q6_k(swiglu_out, W2, NULL, mlp_out, 1, aligned_embed_dim, aligned_intermediate_dim);

    /* Step 10: Final residual add */
    qwen2_0_5b_decode_residual_add_token_major(residual1, mlp_out, output, 1, aligned_embed_dim);
}

/*
 * Layer 10: wq=q5_0 wk=q5_0 wv=q8_0 wo=q5_0 w1=q5_0 w2=q6_k
 */
static void qwen2_0_5b_decode_layer_10_decode(
    QWEN2_0_5B_DECODEModel *model,
    int token_index,
    int aligned_embed_dim,
    int aligned_head_dim,
    int aligned_intermediate_dim,
    int aligned_context_window
) {
    const QWEN2_0_5B_DECODELayerOffsets *L = &QWEN2_0_5B_DECODE_LAYERS[10];

    float *input = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_LAYERS[9].output);

    float *ln1_gamma = QWEN2_0_5B_DECODE_PTR(model, L->ln1_gamma);
    float *ln1_out = QWEN2_0_5B_DECODE_PTR(model, L->ln1_out);
    float *ln2_gamma = QWEN2_0_5B_DECODE_PTR(model, L->ln2_gamma);
    float *ln2_out = QWEN2_0_5B_DECODE_PTR(model, L->ln2_out);
    float *k_cache = QWEN2_0_5B_DECODE_PTR(model, L->k);
    float *v_cache = QWEN2_0_5B_DECODE_PTR(model, L->v);
    float *proj_tmp = QWEN2_0_5B_DECODE_PTR(model, L->proj_tmp);
    float *proj_scratch = QWEN2_0_5B_DECODE_PTR(model, L->proj_scratch);
    float *residual1 = QWEN2_0_5B_DECODE_PTR(model, L->residual1);
    float *mlp_out = QWEN2_0_5B_DECODE_PTR(model, L->mlp_out);
    float *output = QWEN2_0_5B_DECODE_PTR(model, L->output);

    /* Weights (explicit types for layer 10) */
    const void *WQ = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wq);  /* Q5_0 */
    const void *WK = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wk);  /* Q5_0 */
    const void *WV = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wv);  /* Q8_0 */
    const void *WO = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wo);  /* Q5_0 */
    const void *W1 = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->w1);  /* Q5_0 (gate+up) */
    const void *W2 = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->w2);  /* Q6_K (down) */

    /* Attention biases (Qwen2-style) */
    const float *BQ = (const float *)QWEN2_0_5B_DECODE_PTR(model, L->bq);
    const float *BK = (const float *)QWEN2_0_5B_DECODE_PTR(model, L->bk);
    const float *BV = (const float *)QWEN2_0_5B_DECODE_PTR(model, L->bv);

    float *rope_cos = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_GLOBALS.rope_cos_cache);
    float *rope_sin = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_GLOBALS.rope_sin_cache);

    const int H = QWEN2_0_5B_DECODE_NUM_HEADS;
    const int H_kv = QWEN2_0_5B_DECODE_NUM_KV_HEADS;
    const int head_dim = QWEN2_0_5B_DECODE_HEAD_DIM;

    float q_token[H * aligned_head_dim];
    float k_token[H_kv * aligned_head_dim];
    float v_token[H_kv * aligned_head_dim];
    float attn_token[H * aligned_head_dim];

    /* Local MLP buffers (avoid layout dependencies for intermediate values) */
    float fc1_out[2 * aligned_intermediate_dim];
    float swiglu_out[aligned_intermediate_dim];

    /* Step 1: RMSNorm before attention */
    rmsnorm_forward(input,
                    ln1_gamma,
                    ln1_out,
                    NULL,
                    1,
                    QWEN2_0_5B_DECODE_EMBED_DIM,
                    aligned_embed_dim,
                    1e-06f);

    /* Step 2: QKV projection */
    /* Q projection: Q5_0 -> gemm_nt_q5_0 */
    gemm_nt_q5_0(ln1_out, WQ, BQ, q_token, 1, H * head_dim, aligned_embed_dim);

    /* K projection: Q5_0 -> gemm_nt_q5_0 */
    gemm_nt_q5_0(ln1_out, WK, BK, k_token, 1, H_kv * head_dim, aligned_embed_dim);

    /* V projection: Q8_0 -> gemm_nt_q8_0 */
    gemm_nt_q8_0(ln1_out, WV, BV, v_token, 1, H_kv * head_dim, aligned_embed_dim);

    /* Step 3: RoPE */
    rope_forward_qk(q_token,
                    k_token,
                    rope_cos,
                    rope_sin,
                    H,
                    H_kv,
                    1,
                    head_dim,
                    aligned_head_dim,
                    token_index);

    /* Step 4: KV cache write */
    kv_cache_write_head_major(k_token,
                              v_token,
                              k_cache,
                              v_cache,
                              H_kv,
                              token_index,
                              aligned_context_window,
                              head_dim,
                              aligned_head_dim);

    /* Step 5: Attention (decode) */
    attention_forward_decode_head_major_gqa_regular(q_token,
                                                   k_cache,
                                                   v_cache,
                                                   attn_token,
                                                   H,
                                                   H_kv,
                                                   token_index + 1,
                                                   aligned_context_window,
                                                   head_dim,
                                                   aligned_head_dim);

    /* Step 6: Output projection */
    /* WO projection: Q5_0 -> gemm_nt_q5_0 */
    gemm_nt_q5_0(attn_token, WO, NULL, proj_tmp, 1, aligned_embed_dim, H * head_dim);

    /* Step 7: Residual add */
    qwen2_0_5b_decode_residual_add_token_major(input, proj_tmp, residual1, 1, aligned_embed_dim);

    /* Step 8: RMSNorm before MLP */
    rmsnorm_forward(residual1,
                    ln2_gamma,
                    ln2_out,
                    NULL,
                    1,
                    QWEN2_0_5B_DECODE_EMBED_DIM,
                    aligned_embed_dim,
                    1e-06f);

    /* Step 9: MLP (SwiGLU) */
    /* Gate+Up projection: Q5_0 -> gemm_nt_q5_0 */
    gemm_nt_q5_0(ln2_out, W1, NULL, fc1_out, 1, 2 * aligned_intermediate_dim, aligned_embed_dim);

    /* SwiGLU activation */
    swiglu_forward(fc1_out, swiglu_out, 1, aligned_intermediate_dim);

    /* Down projection: Q6_K -> gemm_nt_q6_k */
    gemm_nt_q6_k(swiglu_out, W2, NULL, mlp_out, 1, aligned_embed_dim, aligned_intermediate_dim);

    /* Step 10: Final residual add */
    qwen2_0_5b_decode_residual_add_token_major(residual1, mlp_out, output, 1, aligned_embed_dim);
}

/*
 * Layer 11: wq=q5_0 wk=q5_0 wv=q5_0 wo=q5_0 w1=q5_0 w2=q4_k
 */
static void qwen2_0_5b_decode_layer_11_decode(
    QWEN2_0_5B_DECODEModel *model,
    int token_index,
    int aligned_embed_dim,
    int aligned_head_dim,
    int aligned_intermediate_dim,
    int aligned_context_window
) {
    const QWEN2_0_5B_DECODELayerOffsets *L = &QWEN2_0_5B_DECODE_LAYERS[11];

    float *input = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_LAYERS[10].output);

    float *ln1_gamma = QWEN2_0_5B_DECODE_PTR(model, L->ln1_gamma);
    float *ln1_out = QWEN2_0_5B_DECODE_PTR(model, L->ln1_out);
    float *ln2_gamma = QWEN2_0_5B_DECODE_PTR(model, L->ln2_gamma);
    float *ln2_out = QWEN2_0_5B_DECODE_PTR(model, L->ln2_out);
    float *k_cache = QWEN2_0_5B_DECODE_PTR(model, L->k);
    float *v_cache = QWEN2_0_5B_DECODE_PTR(model, L->v);
    float *proj_tmp = QWEN2_0_5B_DECODE_PTR(model, L->proj_tmp);
    float *proj_scratch = QWEN2_0_5B_DECODE_PTR(model, L->proj_scratch);
    float *residual1 = QWEN2_0_5B_DECODE_PTR(model, L->residual1);
    float *mlp_out = QWEN2_0_5B_DECODE_PTR(model, L->mlp_out);
    float *output = QWEN2_0_5B_DECODE_PTR(model, L->output);

    /* Weights (explicit types for layer 11) */
    const void *WQ = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wq);  /* Q5_0 */
    const void *WK = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wk);  /* Q5_0 */
    const void *WV = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wv);  /* Q5_0 */
    const void *WO = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wo);  /* Q5_0 */
    const void *W1 = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->w1);  /* Q5_0 (gate+up) */
    const void *W2 = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->w2);  /* Q4_K (down) */

    /* Attention biases (Qwen2-style) */
    const float *BQ = (const float *)QWEN2_0_5B_DECODE_PTR(model, L->bq);
    const float *BK = (const float *)QWEN2_0_5B_DECODE_PTR(model, L->bk);
    const float *BV = (const float *)QWEN2_0_5B_DECODE_PTR(model, L->bv);

    float *rope_cos = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_GLOBALS.rope_cos_cache);
    float *rope_sin = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_GLOBALS.rope_sin_cache);

    const int H = QWEN2_0_5B_DECODE_NUM_HEADS;
    const int H_kv = QWEN2_0_5B_DECODE_NUM_KV_HEADS;
    const int head_dim = QWEN2_0_5B_DECODE_HEAD_DIM;

    float q_token[H * aligned_head_dim];
    float k_token[H_kv * aligned_head_dim];
    float v_token[H_kv * aligned_head_dim];
    float attn_token[H * aligned_head_dim];

    /* Local MLP buffers (avoid layout dependencies for intermediate values) */
    float fc1_out[2 * aligned_intermediate_dim];
    float swiglu_out[aligned_intermediate_dim];

    /* Step 1: RMSNorm before attention */
    rmsnorm_forward(input,
                    ln1_gamma,
                    ln1_out,
                    NULL,
                    1,
                    QWEN2_0_5B_DECODE_EMBED_DIM,
                    aligned_embed_dim,
                    1e-06f);

    /* Step 2: QKV projection */
    /* Q projection: Q5_0 -> gemm_nt_q5_0 */
    gemm_nt_q5_0(ln1_out, WQ, BQ, q_token, 1, H * head_dim, aligned_embed_dim);

    /* K projection: Q5_0 -> gemm_nt_q5_0 */
    gemm_nt_q5_0(ln1_out, WK, BK, k_token, 1, H_kv * head_dim, aligned_embed_dim);

    /* V projection: Q5_0 -> gemm_nt_q5_0 */
    gemm_nt_q5_0(ln1_out, WV, BV, v_token, 1, H_kv * head_dim, aligned_embed_dim);

    /* Step 3: RoPE */
    rope_forward_qk(q_token,
                    k_token,
                    rope_cos,
                    rope_sin,
                    H,
                    H_kv,
                    1,
                    head_dim,
                    aligned_head_dim,
                    token_index);

    /* Step 4: KV cache write */
    kv_cache_write_head_major(k_token,
                              v_token,
                              k_cache,
                              v_cache,
                              H_kv,
                              token_index,
                              aligned_context_window,
                              head_dim,
                              aligned_head_dim);

    /* Step 5: Attention (decode) */
    attention_forward_decode_head_major_gqa_regular(q_token,
                                                   k_cache,
                                                   v_cache,
                                                   attn_token,
                                                   H,
                                                   H_kv,
                                                   token_index + 1,
                                                   aligned_context_window,
                                                   head_dim,
                                                   aligned_head_dim);

    /* Step 6: Output projection */
    /* WO projection: Q5_0 -> gemm_nt_q5_0 */
    gemm_nt_q5_0(attn_token, WO, NULL, proj_tmp, 1, aligned_embed_dim, H * head_dim);

    /* Step 7: Residual add */
    qwen2_0_5b_decode_residual_add_token_major(input, proj_tmp, residual1, 1, aligned_embed_dim);

    /* Step 8: RMSNorm before MLP */
    rmsnorm_forward(residual1,
                    ln2_gamma,
                    ln2_out,
                    NULL,
                    1,
                    QWEN2_0_5B_DECODE_EMBED_DIM,
                    aligned_embed_dim,
                    1e-06f);

    /* Step 9: MLP (SwiGLU) */
    /* Gate+Up projection: Q5_0 -> gemm_nt_q5_0 */
    gemm_nt_q5_0(ln2_out, W1, NULL, fc1_out, 1, 2 * aligned_intermediate_dim, aligned_embed_dim);

    /* SwiGLU activation */
    swiglu_forward(fc1_out, swiglu_out, 1, aligned_intermediate_dim);

    /* Down projection: Q4_K -> gemm_nt_q4_k */
    gemm_nt_q4_k(swiglu_out, W2, NULL, mlp_out, 1, aligned_embed_dim, aligned_intermediate_dim);

    /* Step 10: Final residual add */
    qwen2_0_5b_decode_residual_add_token_major(residual1, mlp_out, output, 1, aligned_embed_dim);
}

/*
 * Layer 12: wq=q5_0 wk=q5_0 wv=q5_0 wo=q5_0 w1=q5_0 w2=q4_k
 */
static void qwen2_0_5b_decode_layer_12_decode(
    QWEN2_0_5B_DECODEModel *model,
    int token_index,
    int aligned_embed_dim,
    int aligned_head_dim,
    int aligned_intermediate_dim,
    int aligned_context_window
) {
    const QWEN2_0_5B_DECODELayerOffsets *L = &QWEN2_0_5B_DECODE_LAYERS[12];

    float *input = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_LAYERS[11].output);

    float *ln1_gamma = QWEN2_0_5B_DECODE_PTR(model, L->ln1_gamma);
    float *ln1_out = QWEN2_0_5B_DECODE_PTR(model, L->ln1_out);
    float *ln2_gamma = QWEN2_0_5B_DECODE_PTR(model, L->ln2_gamma);
    float *ln2_out = QWEN2_0_5B_DECODE_PTR(model, L->ln2_out);
    float *k_cache = QWEN2_0_5B_DECODE_PTR(model, L->k);
    float *v_cache = QWEN2_0_5B_DECODE_PTR(model, L->v);
    float *proj_tmp = QWEN2_0_5B_DECODE_PTR(model, L->proj_tmp);
    float *proj_scratch = QWEN2_0_5B_DECODE_PTR(model, L->proj_scratch);
    float *residual1 = QWEN2_0_5B_DECODE_PTR(model, L->residual1);
    float *mlp_out = QWEN2_0_5B_DECODE_PTR(model, L->mlp_out);
    float *output = QWEN2_0_5B_DECODE_PTR(model, L->output);

    /* Weights (explicit types for layer 12) */
    const void *WQ = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wq);  /* Q5_0 */
    const void *WK = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wk);  /* Q5_0 */
    const void *WV = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wv);  /* Q5_0 */
    const void *WO = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wo);  /* Q5_0 */
    const void *W1 = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->w1);  /* Q5_0 (gate+up) */
    const void *W2 = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->w2);  /* Q4_K (down) */

    /* Attention biases (Qwen2-style) */
    const float *BQ = (const float *)QWEN2_0_5B_DECODE_PTR(model, L->bq);
    const float *BK = (const float *)QWEN2_0_5B_DECODE_PTR(model, L->bk);
    const float *BV = (const float *)QWEN2_0_5B_DECODE_PTR(model, L->bv);

    float *rope_cos = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_GLOBALS.rope_cos_cache);
    float *rope_sin = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_GLOBALS.rope_sin_cache);

    const int H = QWEN2_0_5B_DECODE_NUM_HEADS;
    const int H_kv = QWEN2_0_5B_DECODE_NUM_KV_HEADS;
    const int head_dim = QWEN2_0_5B_DECODE_HEAD_DIM;

    float q_token[H * aligned_head_dim];
    float k_token[H_kv * aligned_head_dim];
    float v_token[H_kv * aligned_head_dim];
    float attn_token[H * aligned_head_dim];

    /* Local MLP buffers (avoid layout dependencies for intermediate values) */
    float fc1_out[2 * aligned_intermediate_dim];
    float swiglu_out[aligned_intermediate_dim];

    /* Step 1: RMSNorm before attention */
    rmsnorm_forward(input,
                    ln1_gamma,
                    ln1_out,
                    NULL,
                    1,
                    QWEN2_0_5B_DECODE_EMBED_DIM,
                    aligned_embed_dim,
                    1e-06f);

    /* Step 2: QKV projection */
    /* Q projection: Q5_0 -> gemm_nt_q5_0 */
    gemm_nt_q5_0(ln1_out, WQ, BQ, q_token, 1, H * head_dim, aligned_embed_dim);

    /* K projection: Q5_0 -> gemm_nt_q5_0 */
    gemm_nt_q5_0(ln1_out, WK, BK, k_token, 1, H_kv * head_dim, aligned_embed_dim);

    /* V projection: Q5_0 -> gemm_nt_q5_0 */
    gemm_nt_q5_0(ln1_out, WV, BV, v_token, 1, H_kv * head_dim, aligned_embed_dim);

    /* Step 3: RoPE */
    rope_forward_qk(q_token,
                    k_token,
                    rope_cos,
                    rope_sin,
                    H,
                    H_kv,
                    1,
                    head_dim,
                    aligned_head_dim,
                    token_index);

    /* Step 4: KV cache write */
    kv_cache_write_head_major(k_token,
                              v_token,
                              k_cache,
                              v_cache,
                              H_kv,
                              token_index,
                              aligned_context_window,
                              head_dim,
                              aligned_head_dim);

    /* Step 5: Attention (decode) */
    attention_forward_decode_head_major_gqa_regular(q_token,
                                                   k_cache,
                                                   v_cache,
                                                   attn_token,
                                                   H,
                                                   H_kv,
                                                   token_index + 1,
                                                   aligned_context_window,
                                                   head_dim,
                                                   aligned_head_dim);

    /* Step 6: Output projection */
    /* WO projection: Q5_0 -> gemm_nt_q5_0 */
    gemm_nt_q5_0(attn_token, WO, NULL, proj_tmp, 1, aligned_embed_dim, H * head_dim);

    /* Step 7: Residual add */
    qwen2_0_5b_decode_residual_add_token_major(input, proj_tmp, residual1, 1, aligned_embed_dim);

    /* Step 8: RMSNorm before MLP */
    rmsnorm_forward(residual1,
                    ln2_gamma,
                    ln2_out,
                    NULL,
                    1,
                    QWEN2_0_5B_DECODE_EMBED_DIM,
                    aligned_embed_dim,
                    1e-06f);

    /* Step 9: MLP (SwiGLU) */
    /* Gate+Up projection: Q5_0 -> gemm_nt_q5_0 */
    gemm_nt_q5_0(ln2_out, W1, NULL, fc1_out, 1, 2 * aligned_intermediate_dim, aligned_embed_dim);

    /* SwiGLU activation */
    swiglu_forward(fc1_out, swiglu_out, 1, aligned_intermediate_dim);

    /* Down projection: Q4_K -> gemm_nt_q4_k */
    gemm_nt_q4_k(swiglu_out, W2, NULL, mlp_out, 1, aligned_embed_dim, aligned_intermediate_dim);

    /* Step 10: Final residual add */
    qwen2_0_5b_decode_residual_add_token_major(residual1, mlp_out, output, 1, aligned_embed_dim);
}

/*
 * Layer 13: wq=q5_0 wk=q5_0 wv=q8_0 wo=q5_0 w1=q5_0 w2=q6_k
 */
static void qwen2_0_5b_decode_layer_13_decode(
    QWEN2_0_5B_DECODEModel *model,
    int token_index,
    int aligned_embed_dim,
    int aligned_head_dim,
    int aligned_intermediate_dim,
    int aligned_context_window
) {
    const QWEN2_0_5B_DECODELayerOffsets *L = &QWEN2_0_5B_DECODE_LAYERS[13];

    float *input = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_LAYERS[12].output);

    float *ln1_gamma = QWEN2_0_5B_DECODE_PTR(model, L->ln1_gamma);
    float *ln1_out = QWEN2_0_5B_DECODE_PTR(model, L->ln1_out);
    float *ln2_gamma = QWEN2_0_5B_DECODE_PTR(model, L->ln2_gamma);
    float *ln2_out = QWEN2_0_5B_DECODE_PTR(model, L->ln2_out);
    float *k_cache = QWEN2_0_5B_DECODE_PTR(model, L->k);
    float *v_cache = QWEN2_0_5B_DECODE_PTR(model, L->v);
    float *proj_tmp = QWEN2_0_5B_DECODE_PTR(model, L->proj_tmp);
    float *proj_scratch = QWEN2_0_5B_DECODE_PTR(model, L->proj_scratch);
    float *residual1 = QWEN2_0_5B_DECODE_PTR(model, L->residual1);
    float *mlp_out = QWEN2_0_5B_DECODE_PTR(model, L->mlp_out);
    float *output = QWEN2_0_5B_DECODE_PTR(model, L->output);

    /* Weights (explicit types for layer 13) */
    const void *WQ = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wq);  /* Q5_0 */
    const void *WK = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wk);  /* Q5_0 */
    const void *WV = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wv);  /* Q8_0 */
    const void *WO = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wo);  /* Q5_0 */
    const void *W1 = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->w1);  /* Q5_0 (gate+up) */
    const void *W2 = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->w2);  /* Q6_K (down) */

    /* Attention biases (Qwen2-style) */
    const float *BQ = (const float *)QWEN2_0_5B_DECODE_PTR(model, L->bq);
    const float *BK = (const float *)QWEN2_0_5B_DECODE_PTR(model, L->bk);
    const float *BV = (const float *)QWEN2_0_5B_DECODE_PTR(model, L->bv);

    float *rope_cos = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_GLOBALS.rope_cos_cache);
    float *rope_sin = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_GLOBALS.rope_sin_cache);

    const int H = QWEN2_0_5B_DECODE_NUM_HEADS;
    const int H_kv = QWEN2_0_5B_DECODE_NUM_KV_HEADS;
    const int head_dim = QWEN2_0_5B_DECODE_HEAD_DIM;

    float q_token[H * aligned_head_dim];
    float k_token[H_kv * aligned_head_dim];
    float v_token[H_kv * aligned_head_dim];
    float attn_token[H * aligned_head_dim];

    /* Local MLP buffers (avoid layout dependencies for intermediate values) */
    float fc1_out[2 * aligned_intermediate_dim];
    float swiglu_out[aligned_intermediate_dim];

    /* Step 1: RMSNorm before attention */
    rmsnorm_forward(input,
                    ln1_gamma,
                    ln1_out,
                    NULL,
                    1,
                    QWEN2_0_5B_DECODE_EMBED_DIM,
                    aligned_embed_dim,
                    1e-06f);

    /* Step 2: QKV projection */
    /* Q projection: Q5_0 -> gemm_nt_q5_0 */
    gemm_nt_q5_0(ln1_out, WQ, BQ, q_token, 1, H * head_dim, aligned_embed_dim);

    /* K projection: Q5_0 -> gemm_nt_q5_0 */
    gemm_nt_q5_0(ln1_out, WK, BK, k_token, 1, H_kv * head_dim, aligned_embed_dim);

    /* V projection: Q8_0 -> gemm_nt_q8_0 */
    gemm_nt_q8_0(ln1_out, WV, BV, v_token, 1, H_kv * head_dim, aligned_embed_dim);

    /* Step 3: RoPE */
    rope_forward_qk(q_token,
                    k_token,
                    rope_cos,
                    rope_sin,
                    H,
                    H_kv,
                    1,
                    head_dim,
                    aligned_head_dim,
                    token_index);

    /* Step 4: KV cache write */
    kv_cache_write_head_major(k_token,
                              v_token,
                              k_cache,
                              v_cache,
                              H_kv,
                              token_index,
                              aligned_context_window,
                              head_dim,
                              aligned_head_dim);

    /* Step 5: Attention (decode) */
    attention_forward_decode_head_major_gqa_regular(q_token,
                                                   k_cache,
                                                   v_cache,
                                                   attn_token,
                                                   H,
                                                   H_kv,
                                                   token_index + 1,
                                                   aligned_context_window,
                                                   head_dim,
                                                   aligned_head_dim);

    /* Step 6: Output projection */
    /* WO projection: Q5_0 -> gemm_nt_q5_0 */
    gemm_nt_q5_0(attn_token, WO, NULL, proj_tmp, 1, aligned_embed_dim, H * head_dim);

    /* Step 7: Residual add */
    qwen2_0_5b_decode_residual_add_token_major(input, proj_tmp, residual1, 1, aligned_embed_dim);

    /* Step 8: RMSNorm before MLP */
    rmsnorm_forward(residual1,
                    ln2_gamma,
                    ln2_out,
                    NULL,
                    1,
                    QWEN2_0_5B_DECODE_EMBED_DIM,
                    aligned_embed_dim,
                    1e-06f);

    /* Step 9: MLP (SwiGLU) */
    /* Gate+Up projection: Q5_0 -> gemm_nt_q5_0 */
    gemm_nt_q5_0(ln2_out, W1, NULL, fc1_out, 1, 2 * aligned_intermediate_dim, aligned_embed_dim);

    /* SwiGLU activation */
    swiglu_forward(fc1_out, swiglu_out, 1, aligned_intermediate_dim);

    /* Down projection: Q6_K -> gemm_nt_q6_k */
    gemm_nt_q6_k(swiglu_out, W2, NULL, mlp_out, 1, aligned_embed_dim, aligned_intermediate_dim);

    /* Step 10: Final residual add */
    qwen2_0_5b_decode_residual_add_token_major(residual1, mlp_out, output, 1, aligned_embed_dim);
}

/*
 * Layer 14: wq=q5_0 wk=q5_0 wv=q5_0 wo=q5_0 w1=q5_0 w2=q4_k
 */
static void qwen2_0_5b_decode_layer_14_decode(
    QWEN2_0_5B_DECODEModel *model,
    int token_index,
    int aligned_embed_dim,
    int aligned_head_dim,
    int aligned_intermediate_dim,
    int aligned_context_window
) {
    const QWEN2_0_5B_DECODELayerOffsets *L = &QWEN2_0_5B_DECODE_LAYERS[14];

    float *input = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_LAYERS[13].output);

    float *ln1_gamma = QWEN2_0_5B_DECODE_PTR(model, L->ln1_gamma);
    float *ln1_out = QWEN2_0_5B_DECODE_PTR(model, L->ln1_out);
    float *ln2_gamma = QWEN2_0_5B_DECODE_PTR(model, L->ln2_gamma);
    float *ln2_out = QWEN2_0_5B_DECODE_PTR(model, L->ln2_out);
    float *k_cache = QWEN2_0_5B_DECODE_PTR(model, L->k);
    float *v_cache = QWEN2_0_5B_DECODE_PTR(model, L->v);
    float *proj_tmp = QWEN2_0_5B_DECODE_PTR(model, L->proj_tmp);
    float *proj_scratch = QWEN2_0_5B_DECODE_PTR(model, L->proj_scratch);
    float *residual1 = QWEN2_0_5B_DECODE_PTR(model, L->residual1);
    float *mlp_out = QWEN2_0_5B_DECODE_PTR(model, L->mlp_out);
    float *output = QWEN2_0_5B_DECODE_PTR(model, L->output);

    /* Weights (explicit types for layer 14) */
    const void *WQ = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wq);  /* Q5_0 */
    const void *WK = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wk);  /* Q5_0 */
    const void *WV = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wv);  /* Q5_0 */
    const void *WO = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wo);  /* Q5_0 */
    const void *W1 = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->w1);  /* Q5_0 (gate+up) */
    const void *W2 = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->w2);  /* Q4_K (down) */

    /* Attention biases (Qwen2-style) */
    const float *BQ = (const float *)QWEN2_0_5B_DECODE_PTR(model, L->bq);
    const float *BK = (const float *)QWEN2_0_5B_DECODE_PTR(model, L->bk);
    const float *BV = (const float *)QWEN2_0_5B_DECODE_PTR(model, L->bv);

    float *rope_cos = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_GLOBALS.rope_cos_cache);
    float *rope_sin = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_GLOBALS.rope_sin_cache);

    const int H = QWEN2_0_5B_DECODE_NUM_HEADS;
    const int H_kv = QWEN2_0_5B_DECODE_NUM_KV_HEADS;
    const int head_dim = QWEN2_0_5B_DECODE_HEAD_DIM;

    float q_token[H * aligned_head_dim];
    float k_token[H_kv * aligned_head_dim];
    float v_token[H_kv * aligned_head_dim];
    float attn_token[H * aligned_head_dim];

    /* Local MLP buffers (avoid layout dependencies for intermediate values) */
    float fc1_out[2 * aligned_intermediate_dim];
    float swiglu_out[aligned_intermediate_dim];

    /* Step 1: RMSNorm before attention */
    rmsnorm_forward(input,
                    ln1_gamma,
                    ln1_out,
                    NULL,
                    1,
                    QWEN2_0_5B_DECODE_EMBED_DIM,
                    aligned_embed_dim,
                    1e-06f);

    /* Step 2: QKV projection */
    /* Q projection: Q5_0 -> gemm_nt_q5_0 */
    gemm_nt_q5_0(ln1_out, WQ, BQ, q_token, 1, H * head_dim, aligned_embed_dim);

    /* K projection: Q5_0 -> gemm_nt_q5_0 */
    gemm_nt_q5_0(ln1_out, WK, BK, k_token, 1, H_kv * head_dim, aligned_embed_dim);

    /* V projection: Q5_0 -> gemm_nt_q5_0 */
    gemm_nt_q5_0(ln1_out, WV, BV, v_token, 1, H_kv * head_dim, aligned_embed_dim);

    /* Step 3: RoPE */
    rope_forward_qk(q_token,
                    k_token,
                    rope_cos,
                    rope_sin,
                    H,
                    H_kv,
                    1,
                    head_dim,
                    aligned_head_dim,
                    token_index);

    /* Step 4: KV cache write */
    kv_cache_write_head_major(k_token,
                              v_token,
                              k_cache,
                              v_cache,
                              H_kv,
                              token_index,
                              aligned_context_window,
                              head_dim,
                              aligned_head_dim);

    /* Step 5: Attention (decode) */
    attention_forward_decode_head_major_gqa_regular(q_token,
                                                   k_cache,
                                                   v_cache,
                                                   attn_token,
                                                   H,
                                                   H_kv,
                                                   token_index + 1,
                                                   aligned_context_window,
                                                   head_dim,
                                                   aligned_head_dim);

    /* Step 6: Output projection */
    /* WO projection: Q5_0 -> gemm_nt_q5_0 */
    gemm_nt_q5_0(attn_token, WO, NULL, proj_tmp, 1, aligned_embed_dim, H * head_dim);

    /* Step 7: Residual add */
    qwen2_0_5b_decode_residual_add_token_major(input, proj_tmp, residual1, 1, aligned_embed_dim);

    /* Step 8: RMSNorm before MLP */
    rmsnorm_forward(residual1,
                    ln2_gamma,
                    ln2_out,
                    NULL,
                    1,
                    QWEN2_0_5B_DECODE_EMBED_DIM,
                    aligned_embed_dim,
                    1e-06f);

    /* Step 9: MLP (SwiGLU) */
    /* Gate+Up projection: Q5_0 -> gemm_nt_q5_0 */
    gemm_nt_q5_0(ln2_out, W1, NULL, fc1_out, 1, 2 * aligned_intermediate_dim, aligned_embed_dim);

    /* SwiGLU activation */
    swiglu_forward(fc1_out, swiglu_out, 1, aligned_intermediate_dim);

    /* Down projection: Q4_K -> gemm_nt_q4_k */
    gemm_nt_q4_k(swiglu_out, W2, NULL, mlp_out, 1, aligned_embed_dim, aligned_intermediate_dim);

    /* Step 10: Final residual add */
    qwen2_0_5b_decode_residual_add_token_major(residual1, mlp_out, output, 1, aligned_embed_dim);
}

/*
 * Layer 15: wq=q5_0 wk=q5_0 wv=q5_0 wo=q5_0 w1=q5_0 w2=q4_k
 */
static void qwen2_0_5b_decode_layer_15_decode(
    QWEN2_0_5B_DECODEModel *model,
    int token_index,
    int aligned_embed_dim,
    int aligned_head_dim,
    int aligned_intermediate_dim,
    int aligned_context_window
) {
    const QWEN2_0_5B_DECODELayerOffsets *L = &QWEN2_0_5B_DECODE_LAYERS[15];

    float *input = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_LAYERS[14].output);

    float *ln1_gamma = QWEN2_0_5B_DECODE_PTR(model, L->ln1_gamma);
    float *ln1_out = QWEN2_0_5B_DECODE_PTR(model, L->ln1_out);
    float *ln2_gamma = QWEN2_0_5B_DECODE_PTR(model, L->ln2_gamma);
    float *ln2_out = QWEN2_0_5B_DECODE_PTR(model, L->ln2_out);
    float *k_cache = QWEN2_0_5B_DECODE_PTR(model, L->k);
    float *v_cache = QWEN2_0_5B_DECODE_PTR(model, L->v);
    float *proj_tmp = QWEN2_0_5B_DECODE_PTR(model, L->proj_tmp);
    float *proj_scratch = QWEN2_0_5B_DECODE_PTR(model, L->proj_scratch);
    float *residual1 = QWEN2_0_5B_DECODE_PTR(model, L->residual1);
    float *mlp_out = QWEN2_0_5B_DECODE_PTR(model, L->mlp_out);
    float *output = QWEN2_0_5B_DECODE_PTR(model, L->output);

    /* Weights (explicit types for layer 15) */
    const void *WQ = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wq);  /* Q5_0 */
    const void *WK = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wk);  /* Q5_0 */
    const void *WV = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wv);  /* Q5_0 */
    const void *WO = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wo);  /* Q5_0 */
    const void *W1 = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->w1);  /* Q5_0 (gate+up) */
    const void *W2 = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->w2);  /* Q4_K (down) */

    /* Attention biases (Qwen2-style) */
    const float *BQ = (const float *)QWEN2_0_5B_DECODE_PTR(model, L->bq);
    const float *BK = (const float *)QWEN2_0_5B_DECODE_PTR(model, L->bk);
    const float *BV = (const float *)QWEN2_0_5B_DECODE_PTR(model, L->bv);

    float *rope_cos = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_GLOBALS.rope_cos_cache);
    float *rope_sin = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_GLOBALS.rope_sin_cache);

    const int H = QWEN2_0_5B_DECODE_NUM_HEADS;
    const int H_kv = QWEN2_0_5B_DECODE_NUM_KV_HEADS;
    const int head_dim = QWEN2_0_5B_DECODE_HEAD_DIM;

    float q_token[H * aligned_head_dim];
    float k_token[H_kv * aligned_head_dim];
    float v_token[H_kv * aligned_head_dim];
    float attn_token[H * aligned_head_dim];

    /* Local MLP buffers (avoid layout dependencies for intermediate values) */
    float fc1_out[2 * aligned_intermediate_dim];
    float swiglu_out[aligned_intermediate_dim];

    /* Step 1: RMSNorm before attention */
    rmsnorm_forward(input,
                    ln1_gamma,
                    ln1_out,
                    NULL,
                    1,
                    QWEN2_0_5B_DECODE_EMBED_DIM,
                    aligned_embed_dim,
                    1e-06f);

    /* Step 2: QKV projection */
    /* Q projection: Q5_0 -> gemm_nt_q5_0 */
    gemm_nt_q5_0(ln1_out, WQ, BQ, q_token, 1, H * head_dim, aligned_embed_dim);

    /* K projection: Q5_0 -> gemm_nt_q5_0 */
    gemm_nt_q5_0(ln1_out, WK, BK, k_token, 1, H_kv * head_dim, aligned_embed_dim);

    /* V projection: Q5_0 -> gemm_nt_q5_0 */
    gemm_nt_q5_0(ln1_out, WV, BV, v_token, 1, H_kv * head_dim, aligned_embed_dim);

    /* Step 3: RoPE */
    rope_forward_qk(q_token,
                    k_token,
                    rope_cos,
                    rope_sin,
                    H,
                    H_kv,
                    1,
                    head_dim,
                    aligned_head_dim,
                    token_index);

    /* Step 4: KV cache write */
    kv_cache_write_head_major(k_token,
                              v_token,
                              k_cache,
                              v_cache,
                              H_kv,
                              token_index,
                              aligned_context_window,
                              head_dim,
                              aligned_head_dim);

    /* Step 5: Attention (decode) */
    attention_forward_decode_head_major_gqa_regular(q_token,
                                                   k_cache,
                                                   v_cache,
                                                   attn_token,
                                                   H,
                                                   H_kv,
                                                   token_index + 1,
                                                   aligned_context_window,
                                                   head_dim,
                                                   aligned_head_dim);

    /* Step 6: Output projection */
    /* WO projection: Q5_0 -> gemm_nt_q5_0 */
    gemm_nt_q5_0(attn_token, WO, NULL, proj_tmp, 1, aligned_embed_dim, H * head_dim);

    /* Step 7: Residual add */
    qwen2_0_5b_decode_residual_add_token_major(input, proj_tmp, residual1, 1, aligned_embed_dim);

    /* Step 8: RMSNorm before MLP */
    rmsnorm_forward(residual1,
                    ln2_gamma,
                    ln2_out,
                    NULL,
                    1,
                    QWEN2_0_5B_DECODE_EMBED_DIM,
                    aligned_embed_dim,
                    1e-06f);

    /* Step 9: MLP (SwiGLU) */
    /* Gate+Up projection: Q5_0 -> gemm_nt_q5_0 */
    gemm_nt_q5_0(ln2_out, W1, NULL, fc1_out, 1, 2 * aligned_intermediate_dim, aligned_embed_dim);

    /* SwiGLU activation */
    swiglu_forward(fc1_out, swiglu_out, 1, aligned_intermediate_dim);

    /* Down projection: Q4_K -> gemm_nt_q4_k */
    gemm_nt_q4_k(swiglu_out, W2, NULL, mlp_out, 1, aligned_embed_dim, aligned_intermediate_dim);

    /* Step 10: Final residual add */
    qwen2_0_5b_decode_residual_add_token_major(residual1, mlp_out, output, 1, aligned_embed_dim);
}

/*
 * Layer 16: wq=q5_0 wk=q5_0 wv=q8_0 wo=q5_0 w1=q5_0 w2=q6_k
 */
static void qwen2_0_5b_decode_layer_16_decode(
    QWEN2_0_5B_DECODEModel *model,
    int token_index,
    int aligned_embed_dim,
    int aligned_head_dim,
    int aligned_intermediate_dim,
    int aligned_context_window
) {
    const QWEN2_0_5B_DECODELayerOffsets *L = &QWEN2_0_5B_DECODE_LAYERS[16];

    float *input = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_LAYERS[15].output);

    float *ln1_gamma = QWEN2_0_5B_DECODE_PTR(model, L->ln1_gamma);
    float *ln1_out = QWEN2_0_5B_DECODE_PTR(model, L->ln1_out);
    float *ln2_gamma = QWEN2_0_5B_DECODE_PTR(model, L->ln2_gamma);
    float *ln2_out = QWEN2_0_5B_DECODE_PTR(model, L->ln2_out);
    float *k_cache = QWEN2_0_5B_DECODE_PTR(model, L->k);
    float *v_cache = QWEN2_0_5B_DECODE_PTR(model, L->v);
    float *proj_tmp = QWEN2_0_5B_DECODE_PTR(model, L->proj_tmp);
    float *proj_scratch = QWEN2_0_5B_DECODE_PTR(model, L->proj_scratch);
    float *residual1 = QWEN2_0_5B_DECODE_PTR(model, L->residual1);
    float *mlp_out = QWEN2_0_5B_DECODE_PTR(model, L->mlp_out);
    float *output = QWEN2_0_5B_DECODE_PTR(model, L->output);

    /* Weights (explicit types for layer 16) */
    const void *WQ = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wq);  /* Q5_0 */
    const void *WK = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wk);  /* Q5_0 */
    const void *WV = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wv);  /* Q8_0 */
    const void *WO = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wo);  /* Q5_0 */
    const void *W1 = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->w1);  /* Q5_0 (gate+up) */
    const void *W2 = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->w2);  /* Q6_K (down) */

    /* Attention biases (Qwen2-style) */
    const float *BQ = (const float *)QWEN2_0_5B_DECODE_PTR(model, L->bq);
    const float *BK = (const float *)QWEN2_0_5B_DECODE_PTR(model, L->bk);
    const float *BV = (const float *)QWEN2_0_5B_DECODE_PTR(model, L->bv);

    float *rope_cos = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_GLOBALS.rope_cos_cache);
    float *rope_sin = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_GLOBALS.rope_sin_cache);

    const int H = QWEN2_0_5B_DECODE_NUM_HEADS;
    const int H_kv = QWEN2_0_5B_DECODE_NUM_KV_HEADS;
    const int head_dim = QWEN2_0_5B_DECODE_HEAD_DIM;

    float q_token[H * aligned_head_dim];
    float k_token[H_kv * aligned_head_dim];
    float v_token[H_kv * aligned_head_dim];
    float attn_token[H * aligned_head_dim];

    /* Local MLP buffers (avoid layout dependencies for intermediate values) */
    float fc1_out[2 * aligned_intermediate_dim];
    float swiglu_out[aligned_intermediate_dim];

    /* Step 1: RMSNorm before attention */
    rmsnorm_forward(input,
                    ln1_gamma,
                    ln1_out,
                    NULL,
                    1,
                    QWEN2_0_5B_DECODE_EMBED_DIM,
                    aligned_embed_dim,
                    1e-06f);

    /* Step 2: QKV projection */
    /* Q projection: Q5_0 -> gemm_nt_q5_0 */
    gemm_nt_q5_0(ln1_out, WQ, BQ, q_token, 1, H * head_dim, aligned_embed_dim);

    /* K projection: Q5_0 -> gemm_nt_q5_0 */
    gemm_nt_q5_0(ln1_out, WK, BK, k_token, 1, H_kv * head_dim, aligned_embed_dim);

    /* V projection: Q8_0 -> gemm_nt_q8_0 */
    gemm_nt_q8_0(ln1_out, WV, BV, v_token, 1, H_kv * head_dim, aligned_embed_dim);

    /* Step 3: RoPE */
    rope_forward_qk(q_token,
                    k_token,
                    rope_cos,
                    rope_sin,
                    H,
                    H_kv,
                    1,
                    head_dim,
                    aligned_head_dim,
                    token_index);

    /* Step 4: KV cache write */
    kv_cache_write_head_major(k_token,
                              v_token,
                              k_cache,
                              v_cache,
                              H_kv,
                              token_index,
                              aligned_context_window,
                              head_dim,
                              aligned_head_dim);

    /* Step 5: Attention (decode) */
    attention_forward_decode_head_major_gqa_regular(q_token,
                                                   k_cache,
                                                   v_cache,
                                                   attn_token,
                                                   H,
                                                   H_kv,
                                                   token_index + 1,
                                                   aligned_context_window,
                                                   head_dim,
                                                   aligned_head_dim);

    /* Step 6: Output projection */
    /* WO projection: Q5_0 -> gemm_nt_q5_0 */
    gemm_nt_q5_0(attn_token, WO, NULL, proj_tmp, 1, aligned_embed_dim, H * head_dim);

    /* Step 7: Residual add */
    qwen2_0_5b_decode_residual_add_token_major(input, proj_tmp, residual1, 1, aligned_embed_dim);

    /* Step 8: RMSNorm before MLP */
    rmsnorm_forward(residual1,
                    ln2_gamma,
                    ln2_out,
                    NULL,
                    1,
                    QWEN2_0_5B_DECODE_EMBED_DIM,
                    aligned_embed_dim,
                    1e-06f);

    /* Step 9: MLP (SwiGLU) */
    /* Gate+Up projection: Q5_0 -> gemm_nt_q5_0 */
    gemm_nt_q5_0(ln2_out, W1, NULL, fc1_out, 1, 2 * aligned_intermediate_dim, aligned_embed_dim);

    /* SwiGLU activation */
    swiglu_forward(fc1_out, swiglu_out, 1, aligned_intermediate_dim);

    /* Down projection: Q6_K -> gemm_nt_q6_k */
    gemm_nt_q6_k(swiglu_out, W2, NULL, mlp_out, 1, aligned_embed_dim, aligned_intermediate_dim);

    /* Step 10: Final residual add */
    qwen2_0_5b_decode_residual_add_token_major(residual1, mlp_out, output, 1, aligned_embed_dim);
}

/*
 * Layer 17: wq=q5_0 wk=q5_0 wv=q5_0 wo=q5_0 w1=q5_0 w2=q4_k
 */
static void qwen2_0_5b_decode_layer_17_decode(
    QWEN2_0_5B_DECODEModel *model,
    int token_index,
    int aligned_embed_dim,
    int aligned_head_dim,
    int aligned_intermediate_dim,
    int aligned_context_window
) {
    const QWEN2_0_5B_DECODELayerOffsets *L = &QWEN2_0_5B_DECODE_LAYERS[17];

    float *input = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_LAYERS[16].output);

    float *ln1_gamma = QWEN2_0_5B_DECODE_PTR(model, L->ln1_gamma);
    float *ln1_out = QWEN2_0_5B_DECODE_PTR(model, L->ln1_out);
    float *ln2_gamma = QWEN2_0_5B_DECODE_PTR(model, L->ln2_gamma);
    float *ln2_out = QWEN2_0_5B_DECODE_PTR(model, L->ln2_out);
    float *k_cache = QWEN2_0_5B_DECODE_PTR(model, L->k);
    float *v_cache = QWEN2_0_5B_DECODE_PTR(model, L->v);
    float *proj_tmp = QWEN2_0_5B_DECODE_PTR(model, L->proj_tmp);
    float *proj_scratch = QWEN2_0_5B_DECODE_PTR(model, L->proj_scratch);
    float *residual1 = QWEN2_0_5B_DECODE_PTR(model, L->residual1);
    float *mlp_out = QWEN2_0_5B_DECODE_PTR(model, L->mlp_out);
    float *output = QWEN2_0_5B_DECODE_PTR(model, L->output);

    /* Weights (explicit types for layer 17) */
    const void *WQ = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wq);  /* Q5_0 */
    const void *WK = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wk);  /* Q5_0 */
    const void *WV = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wv);  /* Q5_0 */
    const void *WO = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wo);  /* Q5_0 */
    const void *W1 = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->w1);  /* Q5_0 (gate+up) */
    const void *W2 = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->w2);  /* Q4_K (down) */

    /* Attention biases (Qwen2-style) */
    const float *BQ = (const float *)QWEN2_0_5B_DECODE_PTR(model, L->bq);
    const float *BK = (const float *)QWEN2_0_5B_DECODE_PTR(model, L->bk);
    const float *BV = (const float *)QWEN2_0_5B_DECODE_PTR(model, L->bv);

    float *rope_cos = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_GLOBALS.rope_cos_cache);
    float *rope_sin = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_GLOBALS.rope_sin_cache);

    const int H = QWEN2_0_5B_DECODE_NUM_HEADS;
    const int H_kv = QWEN2_0_5B_DECODE_NUM_KV_HEADS;
    const int head_dim = QWEN2_0_5B_DECODE_HEAD_DIM;

    float q_token[H * aligned_head_dim];
    float k_token[H_kv * aligned_head_dim];
    float v_token[H_kv * aligned_head_dim];
    float attn_token[H * aligned_head_dim];

    /* Local MLP buffers (avoid layout dependencies for intermediate values) */
    float fc1_out[2 * aligned_intermediate_dim];
    float swiglu_out[aligned_intermediate_dim];

    /* Step 1: RMSNorm before attention */
    rmsnorm_forward(input,
                    ln1_gamma,
                    ln1_out,
                    NULL,
                    1,
                    QWEN2_0_5B_DECODE_EMBED_DIM,
                    aligned_embed_dim,
                    1e-06f);

    /* Step 2: QKV projection */
    /* Q projection: Q5_0 -> gemm_nt_q5_0 */
    gemm_nt_q5_0(ln1_out, WQ, BQ, q_token, 1, H * head_dim, aligned_embed_dim);

    /* K projection: Q5_0 -> gemm_nt_q5_0 */
    gemm_nt_q5_0(ln1_out, WK, BK, k_token, 1, H_kv * head_dim, aligned_embed_dim);

    /* V projection: Q5_0 -> gemm_nt_q5_0 */
    gemm_nt_q5_0(ln1_out, WV, BV, v_token, 1, H_kv * head_dim, aligned_embed_dim);

    /* Step 3: RoPE */
    rope_forward_qk(q_token,
                    k_token,
                    rope_cos,
                    rope_sin,
                    H,
                    H_kv,
                    1,
                    head_dim,
                    aligned_head_dim,
                    token_index);

    /* Step 4: KV cache write */
    kv_cache_write_head_major(k_token,
                              v_token,
                              k_cache,
                              v_cache,
                              H_kv,
                              token_index,
                              aligned_context_window,
                              head_dim,
                              aligned_head_dim);

    /* Step 5: Attention (decode) */
    attention_forward_decode_head_major_gqa_regular(q_token,
                                                   k_cache,
                                                   v_cache,
                                                   attn_token,
                                                   H,
                                                   H_kv,
                                                   token_index + 1,
                                                   aligned_context_window,
                                                   head_dim,
                                                   aligned_head_dim);

    /* Step 6: Output projection */
    /* WO projection: Q5_0 -> gemm_nt_q5_0 */
    gemm_nt_q5_0(attn_token, WO, NULL, proj_tmp, 1, aligned_embed_dim, H * head_dim);

    /* Step 7: Residual add */
    qwen2_0_5b_decode_residual_add_token_major(input, proj_tmp, residual1, 1, aligned_embed_dim);

    /* Step 8: RMSNorm before MLP */
    rmsnorm_forward(residual1,
                    ln2_gamma,
                    ln2_out,
                    NULL,
                    1,
                    QWEN2_0_5B_DECODE_EMBED_DIM,
                    aligned_embed_dim,
                    1e-06f);

    /* Step 9: MLP (SwiGLU) */
    /* Gate+Up projection: Q5_0 -> gemm_nt_q5_0 */
    gemm_nt_q5_0(ln2_out, W1, NULL, fc1_out, 1, 2 * aligned_intermediate_dim, aligned_embed_dim);

    /* SwiGLU activation */
    swiglu_forward(fc1_out, swiglu_out, 1, aligned_intermediate_dim);

    /* Down projection: Q4_K -> gemm_nt_q4_k */
    gemm_nt_q4_k(swiglu_out, W2, NULL, mlp_out, 1, aligned_embed_dim, aligned_intermediate_dim);

    /* Step 10: Final residual add */
    qwen2_0_5b_decode_residual_add_token_major(residual1, mlp_out, output, 1, aligned_embed_dim);
}

/*
 * Layer 18: wq=q5_0 wk=q5_0 wv=q5_0 wo=q5_0 w1=q5_0 w2=q4_k
 */
static void qwen2_0_5b_decode_layer_18_decode(
    QWEN2_0_5B_DECODEModel *model,
    int token_index,
    int aligned_embed_dim,
    int aligned_head_dim,
    int aligned_intermediate_dim,
    int aligned_context_window
) {
    const QWEN2_0_5B_DECODELayerOffsets *L = &QWEN2_0_5B_DECODE_LAYERS[18];

    float *input = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_LAYERS[17].output);

    float *ln1_gamma = QWEN2_0_5B_DECODE_PTR(model, L->ln1_gamma);
    float *ln1_out = QWEN2_0_5B_DECODE_PTR(model, L->ln1_out);
    float *ln2_gamma = QWEN2_0_5B_DECODE_PTR(model, L->ln2_gamma);
    float *ln2_out = QWEN2_0_5B_DECODE_PTR(model, L->ln2_out);
    float *k_cache = QWEN2_0_5B_DECODE_PTR(model, L->k);
    float *v_cache = QWEN2_0_5B_DECODE_PTR(model, L->v);
    float *proj_tmp = QWEN2_0_5B_DECODE_PTR(model, L->proj_tmp);
    float *proj_scratch = QWEN2_0_5B_DECODE_PTR(model, L->proj_scratch);
    float *residual1 = QWEN2_0_5B_DECODE_PTR(model, L->residual1);
    float *mlp_out = QWEN2_0_5B_DECODE_PTR(model, L->mlp_out);
    float *output = QWEN2_0_5B_DECODE_PTR(model, L->output);

    /* Weights (explicit types for layer 18) */
    const void *WQ = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wq);  /* Q5_0 */
    const void *WK = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wk);  /* Q5_0 */
    const void *WV = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wv);  /* Q5_0 */
    const void *WO = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wo);  /* Q5_0 */
    const void *W1 = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->w1);  /* Q5_0 (gate+up) */
    const void *W2 = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->w2);  /* Q4_K (down) */

    /* Attention biases (Qwen2-style) */
    const float *BQ = (const float *)QWEN2_0_5B_DECODE_PTR(model, L->bq);
    const float *BK = (const float *)QWEN2_0_5B_DECODE_PTR(model, L->bk);
    const float *BV = (const float *)QWEN2_0_5B_DECODE_PTR(model, L->bv);

    float *rope_cos = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_GLOBALS.rope_cos_cache);
    float *rope_sin = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_GLOBALS.rope_sin_cache);

    const int H = QWEN2_0_5B_DECODE_NUM_HEADS;
    const int H_kv = QWEN2_0_5B_DECODE_NUM_KV_HEADS;
    const int head_dim = QWEN2_0_5B_DECODE_HEAD_DIM;

    float q_token[H * aligned_head_dim];
    float k_token[H_kv * aligned_head_dim];
    float v_token[H_kv * aligned_head_dim];
    float attn_token[H * aligned_head_dim];

    /* Local MLP buffers (avoid layout dependencies for intermediate values) */
    float fc1_out[2 * aligned_intermediate_dim];
    float swiglu_out[aligned_intermediate_dim];

    /* Step 1: RMSNorm before attention */
    rmsnorm_forward(input,
                    ln1_gamma,
                    ln1_out,
                    NULL,
                    1,
                    QWEN2_0_5B_DECODE_EMBED_DIM,
                    aligned_embed_dim,
                    1e-06f);

    /* Step 2: QKV projection */
    /* Q projection: Q5_0 -> gemm_nt_q5_0 */
    gemm_nt_q5_0(ln1_out, WQ, BQ, q_token, 1, H * head_dim, aligned_embed_dim);

    /* K projection: Q5_0 -> gemm_nt_q5_0 */
    gemm_nt_q5_0(ln1_out, WK, BK, k_token, 1, H_kv * head_dim, aligned_embed_dim);

    /* V projection: Q5_0 -> gemm_nt_q5_0 */
    gemm_nt_q5_0(ln1_out, WV, BV, v_token, 1, H_kv * head_dim, aligned_embed_dim);

    /* Step 3: RoPE */
    rope_forward_qk(q_token,
                    k_token,
                    rope_cos,
                    rope_sin,
                    H,
                    H_kv,
                    1,
                    head_dim,
                    aligned_head_dim,
                    token_index);

    /* Step 4: KV cache write */
    kv_cache_write_head_major(k_token,
                              v_token,
                              k_cache,
                              v_cache,
                              H_kv,
                              token_index,
                              aligned_context_window,
                              head_dim,
                              aligned_head_dim);

    /* Step 5: Attention (decode) */
    attention_forward_decode_head_major_gqa_regular(q_token,
                                                   k_cache,
                                                   v_cache,
                                                   attn_token,
                                                   H,
                                                   H_kv,
                                                   token_index + 1,
                                                   aligned_context_window,
                                                   head_dim,
                                                   aligned_head_dim);

    /* Step 6: Output projection */
    /* WO projection: Q5_0 -> gemm_nt_q5_0 */
    gemm_nt_q5_0(attn_token, WO, NULL, proj_tmp, 1, aligned_embed_dim, H * head_dim);

    /* Step 7: Residual add */
    qwen2_0_5b_decode_residual_add_token_major(input, proj_tmp, residual1, 1, aligned_embed_dim);

    /* Step 8: RMSNorm before MLP */
    rmsnorm_forward(residual1,
                    ln2_gamma,
                    ln2_out,
                    NULL,
                    1,
                    QWEN2_0_5B_DECODE_EMBED_DIM,
                    aligned_embed_dim,
                    1e-06f);

    /* Step 9: MLP (SwiGLU) */
    /* Gate+Up projection: Q5_0 -> gemm_nt_q5_0 */
    gemm_nt_q5_0(ln2_out, W1, NULL, fc1_out, 1, 2 * aligned_intermediate_dim, aligned_embed_dim);

    /* SwiGLU activation */
    swiglu_forward(fc1_out, swiglu_out, 1, aligned_intermediate_dim);

    /* Down projection: Q4_K -> gemm_nt_q4_k */
    gemm_nt_q4_k(swiglu_out, W2, NULL, mlp_out, 1, aligned_embed_dim, aligned_intermediate_dim);

    /* Step 10: Final residual add */
    qwen2_0_5b_decode_residual_add_token_major(residual1, mlp_out, output, 1, aligned_embed_dim);
}

/*
 * Layer 19: wq=q5_0 wk=q5_0 wv=q8_0 wo=q5_0 w1=q5_0 w2=q6_k
 */
static void qwen2_0_5b_decode_layer_19_decode(
    QWEN2_0_5B_DECODEModel *model,
    int token_index,
    int aligned_embed_dim,
    int aligned_head_dim,
    int aligned_intermediate_dim,
    int aligned_context_window
) {
    const QWEN2_0_5B_DECODELayerOffsets *L = &QWEN2_0_5B_DECODE_LAYERS[19];

    float *input = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_LAYERS[18].output);

    float *ln1_gamma = QWEN2_0_5B_DECODE_PTR(model, L->ln1_gamma);
    float *ln1_out = QWEN2_0_5B_DECODE_PTR(model, L->ln1_out);
    float *ln2_gamma = QWEN2_0_5B_DECODE_PTR(model, L->ln2_gamma);
    float *ln2_out = QWEN2_0_5B_DECODE_PTR(model, L->ln2_out);
    float *k_cache = QWEN2_0_5B_DECODE_PTR(model, L->k);
    float *v_cache = QWEN2_0_5B_DECODE_PTR(model, L->v);
    float *proj_tmp = QWEN2_0_5B_DECODE_PTR(model, L->proj_tmp);
    float *proj_scratch = QWEN2_0_5B_DECODE_PTR(model, L->proj_scratch);
    float *residual1 = QWEN2_0_5B_DECODE_PTR(model, L->residual1);
    float *mlp_out = QWEN2_0_5B_DECODE_PTR(model, L->mlp_out);
    float *output = QWEN2_0_5B_DECODE_PTR(model, L->output);

    /* Weights (explicit types for layer 19) */
    const void *WQ = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wq);  /* Q5_0 */
    const void *WK = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wk);  /* Q5_0 */
    const void *WV = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wv);  /* Q8_0 */
    const void *WO = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wo);  /* Q5_0 */
    const void *W1 = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->w1);  /* Q5_0 (gate+up) */
    const void *W2 = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->w2);  /* Q6_K (down) */

    /* Attention biases (Qwen2-style) */
    const float *BQ = (const float *)QWEN2_0_5B_DECODE_PTR(model, L->bq);
    const float *BK = (const float *)QWEN2_0_5B_DECODE_PTR(model, L->bk);
    const float *BV = (const float *)QWEN2_0_5B_DECODE_PTR(model, L->bv);

    float *rope_cos = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_GLOBALS.rope_cos_cache);
    float *rope_sin = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_GLOBALS.rope_sin_cache);

    const int H = QWEN2_0_5B_DECODE_NUM_HEADS;
    const int H_kv = QWEN2_0_5B_DECODE_NUM_KV_HEADS;
    const int head_dim = QWEN2_0_5B_DECODE_HEAD_DIM;

    float q_token[H * aligned_head_dim];
    float k_token[H_kv * aligned_head_dim];
    float v_token[H_kv * aligned_head_dim];
    float attn_token[H * aligned_head_dim];

    /* Local MLP buffers (avoid layout dependencies for intermediate values) */
    float fc1_out[2 * aligned_intermediate_dim];
    float swiglu_out[aligned_intermediate_dim];

    /* Step 1: RMSNorm before attention */
    rmsnorm_forward(input,
                    ln1_gamma,
                    ln1_out,
                    NULL,
                    1,
                    QWEN2_0_5B_DECODE_EMBED_DIM,
                    aligned_embed_dim,
                    1e-06f);

    /* Step 2: QKV projection */
    /* Q projection: Q5_0 -> gemm_nt_q5_0 */
    gemm_nt_q5_0(ln1_out, WQ, BQ, q_token, 1, H * head_dim, aligned_embed_dim);

    /* K projection: Q5_0 -> gemm_nt_q5_0 */
    gemm_nt_q5_0(ln1_out, WK, BK, k_token, 1, H_kv * head_dim, aligned_embed_dim);

    /* V projection: Q8_0 -> gemm_nt_q8_0 */
    gemm_nt_q8_0(ln1_out, WV, BV, v_token, 1, H_kv * head_dim, aligned_embed_dim);

    /* Step 3: RoPE */
    rope_forward_qk(q_token,
                    k_token,
                    rope_cos,
                    rope_sin,
                    H,
                    H_kv,
                    1,
                    head_dim,
                    aligned_head_dim,
                    token_index);

    /* Step 4: KV cache write */
    kv_cache_write_head_major(k_token,
                              v_token,
                              k_cache,
                              v_cache,
                              H_kv,
                              token_index,
                              aligned_context_window,
                              head_dim,
                              aligned_head_dim);

    /* Step 5: Attention (decode) */
    attention_forward_decode_head_major_gqa_regular(q_token,
                                                   k_cache,
                                                   v_cache,
                                                   attn_token,
                                                   H,
                                                   H_kv,
                                                   token_index + 1,
                                                   aligned_context_window,
                                                   head_dim,
                                                   aligned_head_dim);

    /* Step 6: Output projection */
    /* WO projection: Q5_0 -> gemm_nt_q5_0 */
    gemm_nt_q5_0(attn_token, WO, NULL, proj_tmp, 1, aligned_embed_dim, H * head_dim);

    /* Step 7: Residual add */
    qwen2_0_5b_decode_residual_add_token_major(input, proj_tmp, residual1, 1, aligned_embed_dim);

    /* Step 8: RMSNorm before MLP */
    rmsnorm_forward(residual1,
                    ln2_gamma,
                    ln2_out,
                    NULL,
                    1,
                    QWEN2_0_5B_DECODE_EMBED_DIM,
                    aligned_embed_dim,
                    1e-06f);

    /* Step 9: MLP (SwiGLU) */
    /* Gate+Up projection: Q5_0 -> gemm_nt_q5_0 */
    gemm_nt_q5_0(ln2_out, W1, NULL, fc1_out, 1, 2 * aligned_intermediate_dim, aligned_embed_dim);

    /* SwiGLU activation */
    swiglu_forward(fc1_out, swiglu_out, 1, aligned_intermediate_dim);

    /* Down projection: Q6_K -> gemm_nt_q6_k */
    gemm_nt_q6_k(swiglu_out, W2, NULL, mlp_out, 1, aligned_embed_dim, aligned_intermediate_dim);

    /* Step 10: Final residual add */
    qwen2_0_5b_decode_residual_add_token_major(residual1, mlp_out, output, 1, aligned_embed_dim);
}

/*
 * Layer 20: wq=q5_0 wk=q5_0 wv=q5_0 wo=q5_0 w1=q5_0 w2=q4_k
 */
static void qwen2_0_5b_decode_layer_20_decode(
    QWEN2_0_5B_DECODEModel *model,
    int token_index,
    int aligned_embed_dim,
    int aligned_head_dim,
    int aligned_intermediate_dim,
    int aligned_context_window
) {
    const QWEN2_0_5B_DECODELayerOffsets *L = &QWEN2_0_5B_DECODE_LAYERS[20];

    float *input = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_LAYERS[19].output);

    float *ln1_gamma = QWEN2_0_5B_DECODE_PTR(model, L->ln1_gamma);
    float *ln1_out = QWEN2_0_5B_DECODE_PTR(model, L->ln1_out);
    float *ln2_gamma = QWEN2_0_5B_DECODE_PTR(model, L->ln2_gamma);
    float *ln2_out = QWEN2_0_5B_DECODE_PTR(model, L->ln2_out);
    float *k_cache = QWEN2_0_5B_DECODE_PTR(model, L->k);
    float *v_cache = QWEN2_0_5B_DECODE_PTR(model, L->v);
    float *proj_tmp = QWEN2_0_5B_DECODE_PTR(model, L->proj_tmp);
    float *proj_scratch = QWEN2_0_5B_DECODE_PTR(model, L->proj_scratch);
    float *residual1 = QWEN2_0_5B_DECODE_PTR(model, L->residual1);
    float *mlp_out = QWEN2_0_5B_DECODE_PTR(model, L->mlp_out);
    float *output = QWEN2_0_5B_DECODE_PTR(model, L->output);

    /* Weights (explicit types for layer 20) */
    const void *WQ = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wq);  /* Q5_0 */
    const void *WK = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wk);  /* Q5_0 */
    const void *WV = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wv);  /* Q5_0 */
    const void *WO = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wo);  /* Q5_0 */
    const void *W1 = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->w1);  /* Q5_0 (gate+up) */
    const void *W2 = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->w2);  /* Q4_K (down) */

    /* Attention biases (Qwen2-style) */
    const float *BQ = (const float *)QWEN2_0_5B_DECODE_PTR(model, L->bq);
    const float *BK = (const float *)QWEN2_0_5B_DECODE_PTR(model, L->bk);
    const float *BV = (const float *)QWEN2_0_5B_DECODE_PTR(model, L->bv);

    float *rope_cos = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_GLOBALS.rope_cos_cache);
    float *rope_sin = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_GLOBALS.rope_sin_cache);

    const int H = QWEN2_0_5B_DECODE_NUM_HEADS;
    const int H_kv = QWEN2_0_5B_DECODE_NUM_KV_HEADS;
    const int head_dim = QWEN2_0_5B_DECODE_HEAD_DIM;

    float q_token[H * aligned_head_dim];
    float k_token[H_kv * aligned_head_dim];
    float v_token[H_kv * aligned_head_dim];
    float attn_token[H * aligned_head_dim];

    /* Local MLP buffers (avoid layout dependencies for intermediate values) */
    float fc1_out[2 * aligned_intermediate_dim];
    float swiglu_out[aligned_intermediate_dim];

    /* Step 1: RMSNorm before attention */
    rmsnorm_forward(input,
                    ln1_gamma,
                    ln1_out,
                    NULL,
                    1,
                    QWEN2_0_5B_DECODE_EMBED_DIM,
                    aligned_embed_dim,
                    1e-06f);

    /* Step 2: QKV projection */
    /* Q projection: Q5_0 -> gemm_nt_q5_0 */
    gemm_nt_q5_0(ln1_out, WQ, BQ, q_token, 1, H * head_dim, aligned_embed_dim);

    /* K projection: Q5_0 -> gemm_nt_q5_0 */
    gemm_nt_q5_0(ln1_out, WK, BK, k_token, 1, H_kv * head_dim, aligned_embed_dim);

    /* V projection: Q5_0 -> gemm_nt_q5_0 */
    gemm_nt_q5_0(ln1_out, WV, BV, v_token, 1, H_kv * head_dim, aligned_embed_dim);

    /* Step 3: RoPE */
    rope_forward_qk(q_token,
                    k_token,
                    rope_cos,
                    rope_sin,
                    H,
                    H_kv,
                    1,
                    head_dim,
                    aligned_head_dim,
                    token_index);

    /* Step 4: KV cache write */
    kv_cache_write_head_major(k_token,
                              v_token,
                              k_cache,
                              v_cache,
                              H_kv,
                              token_index,
                              aligned_context_window,
                              head_dim,
                              aligned_head_dim);

    /* Step 5: Attention (decode) */
    attention_forward_decode_head_major_gqa_regular(q_token,
                                                   k_cache,
                                                   v_cache,
                                                   attn_token,
                                                   H,
                                                   H_kv,
                                                   token_index + 1,
                                                   aligned_context_window,
                                                   head_dim,
                                                   aligned_head_dim);

    /* Step 6: Output projection */
    /* WO projection: Q5_0 -> gemm_nt_q5_0 */
    gemm_nt_q5_0(attn_token, WO, NULL, proj_tmp, 1, aligned_embed_dim, H * head_dim);

    /* Step 7: Residual add */
    qwen2_0_5b_decode_residual_add_token_major(input, proj_tmp, residual1, 1, aligned_embed_dim);

    /* Step 8: RMSNorm before MLP */
    rmsnorm_forward(residual1,
                    ln2_gamma,
                    ln2_out,
                    NULL,
                    1,
                    QWEN2_0_5B_DECODE_EMBED_DIM,
                    aligned_embed_dim,
                    1e-06f);

    /* Step 9: MLP (SwiGLU) */
    /* Gate+Up projection: Q5_0 -> gemm_nt_q5_0 */
    gemm_nt_q5_0(ln2_out, W1, NULL, fc1_out, 1, 2 * aligned_intermediate_dim, aligned_embed_dim);

    /* SwiGLU activation */
    swiglu_forward(fc1_out, swiglu_out, 1, aligned_intermediate_dim);

    /* Down projection: Q4_K -> gemm_nt_q4_k */
    gemm_nt_q4_k(swiglu_out, W2, NULL, mlp_out, 1, aligned_embed_dim, aligned_intermediate_dim);

    /* Step 10: Final residual add */
    qwen2_0_5b_decode_residual_add_token_major(residual1, mlp_out, output, 1, aligned_embed_dim);
}

/*
 * Layer 21: wq=q5_0 wk=q5_0 wv=q8_0 wo=q5_0 w1=q5_0 w2=q6_k
 */
static void qwen2_0_5b_decode_layer_21_decode(
    QWEN2_0_5B_DECODEModel *model,
    int token_index,
    int aligned_embed_dim,
    int aligned_head_dim,
    int aligned_intermediate_dim,
    int aligned_context_window
) {
    const QWEN2_0_5B_DECODELayerOffsets *L = &QWEN2_0_5B_DECODE_LAYERS[21];

    float *input = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_LAYERS[20].output);

    float *ln1_gamma = QWEN2_0_5B_DECODE_PTR(model, L->ln1_gamma);
    float *ln1_out = QWEN2_0_5B_DECODE_PTR(model, L->ln1_out);
    float *ln2_gamma = QWEN2_0_5B_DECODE_PTR(model, L->ln2_gamma);
    float *ln2_out = QWEN2_0_5B_DECODE_PTR(model, L->ln2_out);
    float *k_cache = QWEN2_0_5B_DECODE_PTR(model, L->k);
    float *v_cache = QWEN2_0_5B_DECODE_PTR(model, L->v);
    float *proj_tmp = QWEN2_0_5B_DECODE_PTR(model, L->proj_tmp);
    float *proj_scratch = QWEN2_0_5B_DECODE_PTR(model, L->proj_scratch);
    float *residual1 = QWEN2_0_5B_DECODE_PTR(model, L->residual1);
    float *mlp_out = QWEN2_0_5B_DECODE_PTR(model, L->mlp_out);
    float *output = QWEN2_0_5B_DECODE_PTR(model, L->output);

    /* Weights (explicit types for layer 21) */
    const void *WQ = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wq);  /* Q5_0 */
    const void *WK = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wk);  /* Q5_0 */
    const void *WV = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wv);  /* Q8_0 */
    const void *WO = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wo);  /* Q5_0 */
    const void *W1 = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->w1);  /* Q5_0 (gate+up) */
    const void *W2 = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->w2);  /* Q6_K (down) */

    /* Attention biases (Qwen2-style) */
    const float *BQ = (const float *)QWEN2_0_5B_DECODE_PTR(model, L->bq);
    const float *BK = (const float *)QWEN2_0_5B_DECODE_PTR(model, L->bk);
    const float *BV = (const float *)QWEN2_0_5B_DECODE_PTR(model, L->bv);

    float *rope_cos = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_GLOBALS.rope_cos_cache);
    float *rope_sin = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_GLOBALS.rope_sin_cache);

    const int H = QWEN2_0_5B_DECODE_NUM_HEADS;
    const int H_kv = QWEN2_0_5B_DECODE_NUM_KV_HEADS;
    const int head_dim = QWEN2_0_5B_DECODE_HEAD_DIM;

    float q_token[H * aligned_head_dim];
    float k_token[H_kv * aligned_head_dim];
    float v_token[H_kv * aligned_head_dim];
    float attn_token[H * aligned_head_dim];

    /* Local MLP buffers (avoid layout dependencies for intermediate values) */
    float fc1_out[2 * aligned_intermediate_dim];
    float swiglu_out[aligned_intermediate_dim];

    /* Step 1: RMSNorm before attention */
    rmsnorm_forward(input,
                    ln1_gamma,
                    ln1_out,
                    NULL,
                    1,
                    QWEN2_0_5B_DECODE_EMBED_DIM,
                    aligned_embed_dim,
                    1e-06f);

    /* Step 2: QKV projection */
    /* Q projection: Q5_0 -> gemm_nt_q5_0 */
    gemm_nt_q5_0(ln1_out, WQ, BQ, q_token, 1, H * head_dim, aligned_embed_dim);

    /* K projection: Q5_0 -> gemm_nt_q5_0 */
    gemm_nt_q5_0(ln1_out, WK, BK, k_token, 1, H_kv * head_dim, aligned_embed_dim);

    /* V projection: Q8_0 -> gemm_nt_q8_0 */
    gemm_nt_q8_0(ln1_out, WV, BV, v_token, 1, H_kv * head_dim, aligned_embed_dim);

    /* Step 3: RoPE */
    rope_forward_qk(q_token,
                    k_token,
                    rope_cos,
                    rope_sin,
                    H,
                    H_kv,
                    1,
                    head_dim,
                    aligned_head_dim,
                    token_index);

    /* Step 4: KV cache write */
    kv_cache_write_head_major(k_token,
                              v_token,
                              k_cache,
                              v_cache,
                              H_kv,
                              token_index,
                              aligned_context_window,
                              head_dim,
                              aligned_head_dim);

    /* Step 5: Attention (decode) */
    attention_forward_decode_head_major_gqa_regular(q_token,
                                                   k_cache,
                                                   v_cache,
                                                   attn_token,
                                                   H,
                                                   H_kv,
                                                   token_index + 1,
                                                   aligned_context_window,
                                                   head_dim,
                                                   aligned_head_dim);

    /* Step 6: Output projection */
    /* WO projection: Q5_0 -> gemm_nt_q5_0 */
    gemm_nt_q5_0(attn_token, WO, NULL, proj_tmp, 1, aligned_embed_dim, H * head_dim);

    /* Step 7: Residual add */
    qwen2_0_5b_decode_residual_add_token_major(input, proj_tmp, residual1, 1, aligned_embed_dim);

    /* Step 8: RMSNorm before MLP */
    rmsnorm_forward(residual1,
                    ln2_gamma,
                    ln2_out,
                    NULL,
                    1,
                    QWEN2_0_5B_DECODE_EMBED_DIM,
                    aligned_embed_dim,
                    1e-06f);

    /* Step 9: MLP (SwiGLU) */
    /* Gate+Up projection: Q5_0 -> gemm_nt_q5_0 */
    gemm_nt_q5_0(ln2_out, W1, NULL, fc1_out, 1, 2 * aligned_intermediate_dim, aligned_embed_dim);

    /* SwiGLU activation */
    swiglu_forward(fc1_out, swiglu_out, 1, aligned_intermediate_dim);

    /* Down projection: Q6_K -> gemm_nt_q6_k */
    gemm_nt_q6_k(swiglu_out, W2, NULL, mlp_out, 1, aligned_embed_dim, aligned_intermediate_dim);

    /* Step 10: Final residual add */
    qwen2_0_5b_decode_residual_add_token_major(residual1, mlp_out, output, 1, aligned_embed_dim);
}

/*
 * Layer 22: wq=q5_0 wk=q5_0 wv=q5_0 wo=q5_0 w1=q5_0 w2=q4_k
 */
static void qwen2_0_5b_decode_layer_22_decode(
    QWEN2_0_5B_DECODEModel *model,
    int token_index,
    int aligned_embed_dim,
    int aligned_head_dim,
    int aligned_intermediate_dim,
    int aligned_context_window
) {
    const QWEN2_0_5B_DECODELayerOffsets *L = &QWEN2_0_5B_DECODE_LAYERS[22];

    float *input = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_LAYERS[21].output);

    float *ln1_gamma = QWEN2_0_5B_DECODE_PTR(model, L->ln1_gamma);
    float *ln1_out = QWEN2_0_5B_DECODE_PTR(model, L->ln1_out);
    float *ln2_gamma = QWEN2_0_5B_DECODE_PTR(model, L->ln2_gamma);
    float *ln2_out = QWEN2_0_5B_DECODE_PTR(model, L->ln2_out);
    float *k_cache = QWEN2_0_5B_DECODE_PTR(model, L->k);
    float *v_cache = QWEN2_0_5B_DECODE_PTR(model, L->v);
    float *proj_tmp = QWEN2_0_5B_DECODE_PTR(model, L->proj_tmp);
    float *proj_scratch = QWEN2_0_5B_DECODE_PTR(model, L->proj_scratch);
    float *residual1 = QWEN2_0_5B_DECODE_PTR(model, L->residual1);
    float *mlp_out = QWEN2_0_5B_DECODE_PTR(model, L->mlp_out);
    float *output = QWEN2_0_5B_DECODE_PTR(model, L->output);

    /* Weights (explicit types for layer 22) */
    const void *WQ = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wq);  /* Q5_0 */
    const void *WK = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wk);  /* Q5_0 */
    const void *WV = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wv);  /* Q5_0 */
    const void *WO = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wo);  /* Q5_0 */
    const void *W1 = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->w1);  /* Q5_0 (gate+up) */
    const void *W2 = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->w2);  /* Q4_K (down) */

    /* Attention biases (Qwen2-style) */
    const float *BQ = (const float *)QWEN2_0_5B_DECODE_PTR(model, L->bq);
    const float *BK = (const float *)QWEN2_0_5B_DECODE_PTR(model, L->bk);
    const float *BV = (const float *)QWEN2_0_5B_DECODE_PTR(model, L->bv);

    float *rope_cos = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_GLOBALS.rope_cos_cache);
    float *rope_sin = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_GLOBALS.rope_sin_cache);

    const int H = QWEN2_0_5B_DECODE_NUM_HEADS;
    const int H_kv = QWEN2_0_5B_DECODE_NUM_KV_HEADS;
    const int head_dim = QWEN2_0_5B_DECODE_HEAD_DIM;

    float q_token[H * aligned_head_dim];
    float k_token[H_kv * aligned_head_dim];
    float v_token[H_kv * aligned_head_dim];
    float attn_token[H * aligned_head_dim];

    /* Local MLP buffers (avoid layout dependencies for intermediate values) */
    float fc1_out[2 * aligned_intermediate_dim];
    float swiglu_out[aligned_intermediate_dim];

    /* Step 1: RMSNorm before attention */
    rmsnorm_forward(input,
                    ln1_gamma,
                    ln1_out,
                    NULL,
                    1,
                    QWEN2_0_5B_DECODE_EMBED_DIM,
                    aligned_embed_dim,
                    1e-06f);

    /* Step 2: QKV projection */
    /* Q projection: Q5_0 -> gemm_nt_q5_0 */
    gemm_nt_q5_0(ln1_out, WQ, BQ, q_token, 1, H * head_dim, aligned_embed_dim);

    /* K projection: Q5_0 -> gemm_nt_q5_0 */
    gemm_nt_q5_0(ln1_out, WK, BK, k_token, 1, H_kv * head_dim, aligned_embed_dim);

    /* V projection: Q5_0 -> gemm_nt_q5_0 */
    gemm_nt_q5_0(ln1_out, WV, BV, v_token, 1, H_kv * head_dim, aligned_embed_dim);

    /* Step 3: RoPE */
    rope_forward_qk(q_token,
                    k_token,
                    rope_cos,
                    rope_sin,
                    H,
                    H_kv,
                    1,
                    head_dim,
                    aligned_head_dim,
                    token_index);

    /* Step 4: KV cache write */
    kv_cache_write_head_major(k_token,
                              v_token,
                              k_cache,
                              v_cache,
                              H_kv,
                              token_index,
                              aligned_context_window,
                              head_dim,
                              aligned_head_dim);

    /* Step 5: Attention (decode) */
    attention_forward_decode_head_major_gqa_regular(q_token,
                                                   k_cache,
                                                   v_cache,
                                                   attn_token,
                                                   H,
                                                   H_kv,
                                                   token_index + 1,
                                                   aligned_context_window,
                                                   head_dim,
                                                   aligned_head_dim);

    /* Step 6: Output projection */
    /* WO projection: Q5_0 -> gemm_nt_q5_0 */
    gemm_nt_q5_0(attn_token, WO, NULL, proj_tmp, 1, aligned_embed_dim, H * head_dim);

    /* Step 7: Residual add */
    qwen2_0_5b_decode_residual_add_token_major(input, proj_tmp, residual1, 1, aligned_embed_dim);

    /* Step 8: RMSNorm before MLP */
    rmsnorm_forward(residual1,
                    ln2_gamma,
                    ln2_out,
                    NULL,
                    1,
                    QWEN2_0_5B_DECODE_EMBED_DIM,
                    aligned_embed_dim,
                    1e-06f);

    /* Step 9: MLP (SwiGLU) */
    /* Gate+Up projection: Q5_0 -> gemm_nt_q5_0 */
    gemm_nt_q5_0(ln2_out, W1, NULL, fc1_out, 1, 2 * aligned_intermediate_dim, aligned_embed_dim);

    /* SwiGLU activation */
    swiglu_forward(fc1_out, swiglu_out, 1, aligned_intermediate_dim);

    /* Down projection: Q4_K -> gemm_nt_q4_k */
    gemm_nt_q4_k(swiglu_out, W2, NULL, mlp_out, 1, aligned_embed_dim, aligned_intermediate_dim);

    /* Step 10: Final residual add */
    qwen2_0_5b_decode_residual_add_token_major(residual1, mlp_out, output, 1, aligned_embed_dim);
}

/*
 * Layer 23: wq=q5_0 wk=q5_0 wv=q5_0 wo=q5_0 w1=q5_0 w2=q4_k
 */
static void qwen2_0_5b_decode_layer_23_decode(
    QWEN2_0_5B_DECODEModel *model,
    int token_index,
    int aligned_embed_dim,
    int aligned_head_dim,
    int aligned_intermediate_dim,
    int aligned_context_window
) {
    const QWEN2_0_5B_DECODELayerOffsets *L = &QWEN2_0_5B_DECODE_LAYERS[23];

    float *input = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_LAYERS[22].output);

    float *ln1_gamma = QWEN2_0_5B_DECODE_PTR(model, L->ln1_gamma);
    float *ln1_out = QWEN2_0_5B_DECODE_PTR(model, L->ln1_out);
    float *ln2_gamma = QWEN2_0_5B_DECODE_PTR(model, L->ln2_gamma);
    float *ln2_out = QWEN2_0_5B_DECODE_PTR(model, L->ln2_out);
    float *k_cache = QWEN2_0_5B_DECODE_PTR(model, L->k);
    float *v_cache = QWEN2_0_5B_DECODE_PTR(model, L->v);
    float *proj_tmp = QWEN2_0_5B_DECODE_PTR(model, L->proj_tmp);
    float *proj_scratch = QWEN2_0_5B_DECODE_PTR(model, L->proj_scratch);
    float *residual1 = QWEN2_0_5B_DECODE_PTR(model, L->residual1);
    float *mlp_out = QWEN2_0_5B_DECODE_PTR(model, L->mlp_out);
    float *output = QWEN2_0_5B_DECODE_PTR(model, L->output);

    /* Weights (explicit types for layer 23) */
    const void *WQ = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wq);  /* Q5_0 */
    const void *WK = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wk);  /* Q5_0 */
    const void *WV = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wv);  /* Q5_0 */
    const void *WO = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->wo);  /* Q5_0 */
    const void *W1 = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->w1);  /* Q5_0 (gate+up) */
    const void *W2 = (const void *)QWEN2_0_5B_DECODE_PTR(model, L->w2);  /* Q4_K (down) */

    /* Attention biases (Qwen2-style) */
    const float *BQ = (const float *)QWEN2_0_5B_DECODE_PTR(model, L->bq);
    const float *BK = (const float *)QWEN2_0_5B_DECODE_PTR(model, L->bk);
    const float *BV = (const float *)QWEN2_0_5B_DECODE_PTR(model, L->bv);

    float *rope_cos = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_GLOBALS.rope_cos_cache);
    float *rope_sin = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_GLOBALS.rope_sin_cache);

    const int H = QWEN2_0_5B_DECODE_NUM_HEADS;
    const int H_kv = QWEN2_0_5B_DECODE_NUM_KV_HEADS;
    const int head_dim = QWEN2_0_5B_DECODE_HEAD_DIM;

    float q_token[H * aligned_head_dim];
    float k_token[H_kv * aligned_head_dim];
    float v_token[H_kv * aligned_head_dim];
    float attn_token[H * aligned_head_dim];

    /* Local MLP buffers (avoid layout dependencies for intermediate values) */
    float fc1_out[2 * aligned_intermediate_dim];
    float swiglu_out[aligned_intermediate_dim];

    /* Step 1: RMSNorm before attention */
    rmsnorm_forward(input,
                    ln1_gamma,
                    ln1_out,
                    NULL,
                    1,
                    QWEN2_0_5B_DECODE_EMBED_DIM,
                    aligned_embed_dim,
                    1e-06f);

    /* Step 2: QKV projection */
    /* Q projection: Q5_0 -> gemm_nt_q5_0 */
    gemm_nt_q5_0(ln1_out, WQ, BQ, q_token, 1, H * head_dim, aligned_embed_dim);

    /* K projection: Q5_0 -> gemm_nt_q5_0 */
    gemm_nt_q5_0(ln1_out, WK, BK, k_token, 1, H_kv * head_dim, aligned_embed_dim);

    /* V projection: Q5_0 -> gemm_nt_q5_0 */
    gemm_nt_q5_0(ln1_out, WV, BV, v_token, 1, H_kv * head_dim, aligned_embed_dim);

    /* Step 3: RoPE */
    rope_forward_qk(q_token,
                    k_token,
                    rope_cos,
                    rope_sin,
                    H,
                    H_kv,
                    1,
                    head_dim,
                    aligned_head_dim,
                    token_index);

    /* Step 4: KV cache write */
    kv_cache_write_head_major(k_token,
                              v_token,
                              k_cache,
                              v_cache,
                              H_kv,
                              token_index,
                              aligned_context_window,
                              head_dim,
                              aligned_head_dim);

    /* Step 5: Attention (decode) */
    attention_forward_decode_head_major_gqa_regular(q_token,
                                                   k_cache,
                                                   v_cache,
                                                   attn_token,
                                                   H,
                                                   H_kv,
                                                   token_index + 1,
                                                   aligned_context_window,
                                                   head_dim,
                                                   aligned_head_dim);

    /* Step 6: Output projection */
    /* WO projection: Q5_0 -> gemm_nt_q5_0 */
    gemm_nt_q5_0(attn_token, WO, NULL, proj_tmp, 1, aligned_embed_dim, H * head_dim);

    /* Step 7: Residual add */
    qwen2_0_5b_decode_residual_add_token_major(input, proj_tmp, residual1, 1, aligned_embed_dim);

    /* Step 8: RMSNorm before MLP */
    rmsnorm_forward(residual1,
                    ln2_gamma,
                    ln2_out,
                    NULL,
                    1,
                    QWEN2_0_5B_DECODE_EMBED_DIM,
                    aligned_embed_dim,
                    1e-06f);

    /* Step 9: MLP (SwiGLU) */
    /* Gate+Up projection: Q5_0 -> gemm_nt_q5_0 */
    gemm_nt_q5_0(ln2_out, W1, NULL, fc1_out, 1, 2 * aligned_intermediate_dim, aligned_embed_dim);

    /* SwiGLU activation */
    swiglu_forward(fc1_out, swiglu_out, 1, aligned_intermediate_dim);

    /* Down projection: Q4_K -> gemm_nt_q4_k */
    gemm_nt_q4_k(swiglu_out, W2, NULL, mlp_out, 1, aligned_embed_dim, aligned_intermediate_dim);

    /* Step 10: Final residual add */
    qwen2_0_5b_decode_residual_add_token_major(residual1, mlp_out, output, 1, aligned_embed_dim);
}

/* ============================================================================
 * DECODE TOKEN (calls each layer explicitly)
 * ============================================================================ */

static void qwen2_0_5b_decode_decode_token(
    QWEN2_0_5B_DECODEModel *model,
    const int *token,
    int token_index
) {
    if (!model || !token) return;

    const int aligned_embed_dim = 896;
    const int aligned_head_dim = 64;
    const int aligned_intermediate_dim = 4864;
    const int aligned_context_window = 131072;

    if (token_index < 0 || token_index >= aligned_context_window) return;

    /* Embedding lookup */
    float *embed_out = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_HEADER.embedded_input);
    const void *embed_weight = (const void *)QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_HEADER.token_emb);
    /* Embedding: Q8_0 -> embedding_forward_q8_0 */
    embedding_forward_q8_0((const int32_t *)token,
                          1,
                          QWEN2_0_5B_DECODE_VOCAB_SIZE,
                          embed_weight,
                          NULL,
                          embed_out,
                          QWEN2_0_5B_DECODE_EMBED_DIM,
                          aligned_embed_dim,
                          1,
                          0);

    /* Process each layer explicitly */
    qwen2_0_5b_decode_layer_0_decode(model, token_index, aligned_embed_dim, aligned_head_dim, aligned_intermediate_dim, aligned_context_window);
    qwen2_0_5b_decode_layer_1_decode(model, token_index, aligned_embed_dim, aligned_head_dim, aligned_intermediate_dim, aligned_context_window);
    qwen2_0_5b_decode_layer_2_decode(model, token_index, aligned_embed_dim, aligned_head_dim, aligned_intermediate_dim, aligned_context_window);
    qwen2_0_5b_decode_layer_3_decode(model, token_index, aligned_embed_dim, aligned_head_dim, aligned_intermediate_dim, aligned_context_window);
    qwen2_0_5b_decode_layer_4_decode(model, token_index, aligned_embed_dim, aligned_head_dim, aligned_intermediate_dim, aligned_context_window);
    qwen2_0_5b_decode_layer_5_decode(model, token_index, aligned_embed_dim, aligned_head_dim, aligned_intermediate_dim, aligned_context_window);
    qwen2_0_5b_decode_layer_6_decode(model, token_index, aligned_embed_dim, aligned_head_dim, aligned_intermediate_dim, aligned_context_window);
    qwen2_0_5b_decode_layer_7_decode(model, token_index, aligned_embed_dim, aligned_head_dim, aligned_intermediate_dim, aligned_context_window);
    qwen2_0_5b_decode_layer_8_decode(model, token_index, aligned_embed_dim, aligned_head_dim, aligned_intermediate_dim, aligned_context_window);
    qwen2_0_5b_decode_layer_9_decode(model, token_index, aligned_embed_dim, aligned_head_dim, aligned_intermediate_dim, aligned_context_window);
    qwen2_0_5b_decode_layer_10_decode(model, token_index, aligned_embed_dim, aligned_head_dim, aligned_intermediate_dim, aligned_context_window);
    qwen2_0_5b_decode_layer_11_decode(model, token_index, aligned_embed_dim, aligned_head_dim, aligned_intermediate_dim, aligned_context_window);
    qwen2_0_5b_decode_layer_12_decode(model, token_index, aligned_embed_dim, aligned_head_dim, aligned_intermediate_dim, aligned_context_window);
    qwen2_0_5b_decode_layer_13_decode(model, token_index, aligned_embed_dim, aligned_head_dim, aligned_intermediate_dim, aligned_context_window);
    qwen2_0_5b_decode_layer_14_decode(model, token_index, aligned_embed_dim, aligned_head_dim, aligned_intermediate_dim, aligned_context_window);
    qwen2_0_5b_decode_layer_15_decode(model, token_index, aligned_embed_dim, aligned_head_dim, aligned_intermediate_dim, aligned_context_window);
    qwen2_0_5b_decode_layer_16_decode(model, token_index, aligned_embed_dim, aligned_head_dim, aligned_intermediate_dim, aligned_context_window);
    qwen2_0_5b_decode_layer_17_decode(model, token_index, aligned_embed_dim, aligned_head_dim, aligned_intermediate_dim, aligned_context_window);
    qwen2_0_5b_decode_layer_18_decode(model, token_index, aligned_embed_dim, aligned_head_dim, aligned_intermediate_dim, aligned_context_window);
    qwen2_0_5b_decode_layer_19_decode(model, token_index, aligned_embed_dim, aligned_head_dim, aligned_intermediate_dim, aligned_context_window);
    qwen2_0_5b_decode_layer_20_decode(model, token_index, aligned_embed_dim, aligned_head_dim, aligned_intermediate_dim, aligned_context_window);
    qwen2_0_5b_decode_layer_21_decode(model, token_index, aligned_embed_dim, aligned_head_dim, aligned_intermediate_dim, aligned_context_window);
    qwen2_0_5b_decode_layer_22_decode(model, token_index, aligned_embed_dim, aligned_head_dim, aligned_intermediate_dim, aligned_context_window);
    qwen2_0_5b_decode_layer_23_decode(model, token_index, aligned_embed_dim, aligned_head_dim, aligned_intermediate_dim, aligned_context_window);

    /* Final RMSNorm */
    float *last_hidden = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_LAYERS[23].output);
    float *final_ln_weight = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_FOOTER.final_ln_weight);
    float *final_out = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_FOOTER.final_output);
    rmsnorm_forward(last_hidden,
                    final_ln_weight,
                    final_out,
                    NULL,
                    1,
                    QWEN2_0_5B_DECODE_EMBED_DIM,
                    aligned_embed_dim,
                    1e-06f);

    /* LM head projection */
    float *logits = QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_FOOTER.logits);
    const void *lm_head = (const void *)QWEN2_0_5B_DECODE_PTR(model, QWEN2_0_5B_DECODE_FOOTER.lm_head_weight);
    /* LM head: Q8_0 -> gemm_nt_q8_0 */
    gemm_nt_q8_0(final_out, lm_head, NULL, logits, 1, QWEN2_0_5B_DECODE_VOCAB_SIZE, aligned_embed_dim);
}

/* ============================================================================
 * PUBLIC API
 * ============================================================================ */

void qwen2_0_5b_decode_forward(
    QWEN2_0_5B_DECODEModel *model,
    const int *tokens,
    int num_tokens
) {
    if (!model || !tokens || num_tokens <= 0) return;
    qwen2_0_5b_decode_forward_prefill_impl(model, tokens, num_tokens);
}

void qwen2_0_5b_decode_decode(QWEN2_0_5B_DECODEModel *model, const int *token, int token_index) {
    qwen2_0_5b_decode_decode_token(model, token, token_index);
}

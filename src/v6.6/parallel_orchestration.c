/**
 * @file parallel_orchestration.c
 * @brief [LEGACY] Parallel decode orchestration prototype — NOT USED by v6.6
 *
 * This file was an early prototype demonstrating llama.cpp-style OpenMP
 * parallelization patterns. It is NOT compiled into the v6.6 build and
 * has no callers in the generated inference code path.
 *
 * v6.6 decode runs entirely through the generated code in:
 *   version/v6.6/src/generated/ck-kernel-inference.c
 *     → ck_model_decode_internal()
 *
 * Threading for v6.6 is handled by ck_threadpool (include/ck_threadpool.h),
 * which replaces the OpenMP approach used here.
 *
 * Kept for reference only. See the original design notes below.
 *
 * Original design (OpenMP, superseded):
 * - OpenMP parallel region at orchestration level
 * - Each kernel receives (ith, nth) and processes its slice
 * - Barriers between dependent operations
 * - Key insight: amortize thread pool overhead over entire forward pass
 */

#include <omp.h>
#include <string.h>
#include <math.h>

#include "ckernel_engine.h"
#include "ckernel_quant.h"

/* ============================================================================
 * PARALLEL KERNEL WRAPPERS
 *
 * These call the _parallel_simd versions with thread indices from OpenMP.
 * Each wrapper receives (ith, nth) from the calling parallel region.
 * ============================================================================ */

/**
 * Single-token decode with parallel SIMD kernels.
 *
 * This is the main decode function that processes one token through all layers.
 * OpenMP parallel region is created ONCE at the top, and all kernels
 * receive (ith, nth) to split their work.
 *
 * Pattern:
 *   #pragma omp parallel
 *   {
 *       int ith = omp_get_thread_num();
 *       int nth = omp_get_num_threads();
 *
 *       // Each kernel processes only its slice
 *       gemv_q4_k_q8_k_parallel_simd(..., ith, nth);
 *       #pragma omp barrier
 *
 *       rmsnorm_parallel(..., ith, nth);  // (not implemented yet)
 *       #pragma omp barrier
 *       ...
 *   }
 */

/* Parallel residual add: out = a + b, split across threads */
static void residual_add_parallel(const float *a, const float *b,
                                   float *out, int n,
                                   int ith, int nth)
{
    const int dr = (n + nth - 1) / nth;
    const int r0 = dr * ith;
    const int r1 = (r0 + dr < n) ? (r0 + dr) : n;

    if (r0 >= n) return;

    for (int i = r0; i < r1; i++) {
        out[i] = a[i] + b[i];
    }
}

/* Parallel scale: y[i] *= scale, split across threads */
static void vec_scale_parallel(float *y, float scale, int n,
                                int ith, int nth)
{
    const int dr = (n + nth - 1) / nth;
    const int r0 = dr * ith;
    const int r1 = (r0 + dr < n) ? (r0 + dr) : n;

    if (r0 >= n) return;

    for (int i = r0; i < r1; i++) {
        y[i] *= scale;
    }
}

/* Parallel zero: memset to 0, split across threads */
static void vec_zero_parallel(float *y, int n, int ith, int nth)
{
    const int dr = (n + nth - 1) / nth;
    const int r0 = dr * ith;
    const int r1 = (r0 + dr < n) ? (r0 + dr) : n;

    if (r0 >= n) return;

    memset(y + r0, 0, (r1 - r0) * sizeof(float));
}

/* ============================================================================
 * EXAMPLE: Parallel Q/K/V Projection
 *
 * This shows how to parallelize the QKV projections in one OpenMP region.
 * In practice, this would be integrated into the full decode function.
 * ============================================================================ */

/**
 * Parallel Q/K/V projection for single token decode.
 *
 * @param ln1_q8      Input: RMSNorm output quantized to Q8_K [aligned_embed]
 * @param WQ          Q weights [H*head_dim, aligned_embed] in Q4_K
 * @param WK          K weights [H_kv*head_dim, aligned_embed] in Q4_K
 * @param WV          V weights [H_kv*head_dim, aligned_embed] in Q4_K
 * @param q_out       Output: Q vectors [H, head_dim]
 * @param k_out       Output: K vector [H_kv, head_dim]
 * @param v_out       Output: V vector [H_kv, head_dim]
 * @param H           Number of query heads
 * @param H_kv        Number of KV heads (GQA)
 * @param head_dim    Head dimension
 * @param embed_dim   Embedding dimension
 * @param num_threads Number of threads to use (0 = auto)
 */
void qkv_projection_parallel(const void *ln1_q8,
                              const void *WQ,
                              const void *WK,
                              const void *WV,
                              float *q_out,
                              float *k_out,
                              float *v_out,
                              int H, int H_kv,
                              int head_dim, int embed_dim,
                              int num_threads)
{
    const int q_dim = H * head_dim;
    const int kv_dim = H_kv * head_dim;

    /* Align to QK_K for quantized matmul */
    const int aligned_embed = ((embed_dim + 255) / 256) * 256;

    if (num_threads <= 0) {
        num_threads = omp_get_max_threads();
    }

    /* Single OpenMP region for all three projections */
    #pragma omp parallel num_threads(num_threads)
    {
        const int ith = omp_get_thread_num();
        const int nth = omp_get_num_threads();

        /* Q projection: largest, benefits most from parallelism */
        gemv_q4_k_q8_k_parallel_simd(q_out, WQ, ln1_q8, q_dim, aligned_embed, ith, nth);

        /* K projection: smaller, but still benefits */
        gemv_q4_k_q8_k_parallel_simd(k_out, WK, ln1_q8, kv_dim, aligned_embed, ith, nth);

        /* V projection */
        gemv_q4_k_q8_k_parallel_simd(v_out, WV, ln1_q8, kv_dim, aligned_embed, ith, nth);

        /* Implicit barrier at end of parallel region */
    }
}

/**
 * Parallel MLP (gate/up + SwiGLU + down projection).
 *
 * @param ln2_q8      Input: RMSNorm output quantized to Q8_K
 * @param W_gate      Gate weights [intermediate, embed] in Q4_K
 * @param W_up        Up weights [intermediate, embed] in Q4_K
 * @param W_down      Down weights [embed, intermediate] in Q4_K
 * @param gate_buf    Scratch: gate output [intermediate]
 * @param up_buf      Scratch: up output [intermediate]
 * @param swiglu_buf  Scratch: SwiGLU output [intermediate]
 * @param down_q8     Scratch: down input quantized [intermediate Q8_K blocks]
 * @param mlp_out     Output: MLP output [embed]
 * @param intermediate  Intermediate dimension
 * @param embed_dim   Embedding dimension
 * @param num_threads Number of threads (0 = auto)
 */
void mlp_parallel(const void *ln2_q8,
                   const void *W_gate,
                   const void *W_up,
                   const void *W_down,
                   float *gate_buf,
                   float *up_buf,
                   float *swiglu_buf,
                   void *down_q8,
                   float *mlp_out,
                   int intermediate,
                   int embed_dim,
                   int num_threads)
{
    const int aligned_embed = ((embed_dim + 255) / 256) * 256;
    const int aligned_inter = ((intermediate + 255) / 256) * 256;

    if (num_threads <= 0) {
        num_threads = omp_get_max_threads();
    }

    #pragma omp parallel num_threads(num_threads)
    {
        const int ith = omp_get_thread_num();
        const int nth = omp_get_num_threads();

        /* Gate and Up projections (can run in parallel, no dependency) */
        gemv_q4_k_q8_k_parallel_simd(gate_buf, W_gate, ln2_q8, aligned_inter, aligned_embed, ith, nth);
        gemv_q4_k_q8_k_parallel_simd(up_buf, W_up, ln2_q8, aligned_inter, aligned_embed, ith, nth);

        #pragma omp barrier  /* Wait for gate and up to complete */

        /* SwiGLU: swiglu = silu(gate) * up */
        /* This is element-wise, parallelize across elements */
        const int dr = (aligned_inter + nth - 1) / nth;
        const int r0 = dr * ith;
        const int r1 = (r0 + dr < aligned_inter) ? (r0 + dr) : aligned_inter;

        for (int i = r0; i < r1 && i < intermediate; i++) {
            float g = gate_buf[i];
            float silu_g = g / (1.0f + expf(-g));  /* SiLU activation */
            swiglu_buf[i] = silu_g * up_buf[i];
        }

        #pragma omp barrier  /* Wait for SwiGLU to complete */

        /* Down projection: only thread 0 quantizes (single-threaded) */
        /* TODO: Add parallel quantization */
        #pragma omp single
        {
            quantize_row_q8_k(swiglu_buf, down_q8, aligned_inter);
        }
        /* Implicit barrier after omp single */

        /* Down projection */
        gemv_q4_k_q8_k_parallel_simd(mlp_out, W_down, down_q8, aligned_embed, aligned_inter, ith, nth);
    }
}

/* ============================================================================
 * FULL LAYER DECODE (parallel)
 *
 * Processes one transformer layer with all operations parallelized.
 * ============================================================================ */

/**
 * Process one transformer layer in parallel.
 *
 * This demonstrates the full parallel pattern for a single layer.
 * In production, this would be called in a loop for all layers.
 */
void decode_layer_parallel(
    /* Inputs */
    float *hidden,           /* [embed_dim] - modified in place */
    const void *ln1_weight,  /* RMSNorm weights */
    const void *ln2_weight,  /* RMSNorm weights */
    const void *WQ,          /* Q4_K weights */
    const void *WK,
    const void *WV,
    const void *WO,
    const void *W_gate,
    const void *W_up,
    const void *W_down,
    /* KV cache */
    float *k_cache,          /* [H_kv, max_seq, head_dim] */
    float *v_cache,
    int token_index,         /* Current position in sequence */
    /* Scratch buffers */
    float *scratch,          /* Aligned scratch space */
    /* Model config */
    int embed_dim,
    int intermediate,
    int H, int H_kv,
    int head_dim,
    int max_seq,
    float eps,               /* RMSNorm epsilon */
    /* Threading */
    int num_threads)
{
    if (num_threads <= 0) {
        num_threads = omp_get_max_threads();
    }

    /* Align dimensions */
    const int aligned_embed = ((embed_dim + 255) / 256) * 256;
    const int aligned_inter = ((intermediate + 255) / 256) * 256;
    const int aligned_head = ((head_dim + 31) / 32) * 32;

    /* Partition scratch buffer */
    float *ln1_out = scratch;
    float *q_vec = ln1_out + aligned_embed;
    float *k_vec = q_vec + H * aligned_head;
    float *v_vec = k_vec + H_kv * aligned_head;
    float *attn_out = v_vec + H_kv * aligned_head;
    float *o_out = attn_out + H * aligned_head;
    float *ln2_out = o_out + aligned_embed;
    float *gate_buf = ln2_out + aligned_embed;
    float *up_buf = gate_buf + aligned_inter;
    float *swiglu_buf = up_buf + aligned_inter;
    float *mlp_out = swiglu_buf + aligned_inter;

    /* Q8_K buffers for quantized input */
    const size_t q8_embed_bytes = ((aligned_embed + 255) / 256) * 292;
    const size_t q8_inter_bytes = ((aligned_inter + 255) / 256) * 292;
    uint8_t *ln1_q8 = (uint8_t *)(mlp_out + aligned_embed);
    uint8_t *ln2_q8 = ln1_q8 + q8_embed_bytes;
    uint8_t *down_q8 = ln2_q8 + q8_embed_bytes;

    #pragma omp parallel num_threads(num_threads)
    {
        const int ith = omp_get_thread_num();
        const int nth = omp_get_num_threads();

        /* ============================================================
         * ATTENTION BLOCK
         * ============================================================ */

        /* Step 1: RMSNorm (TODO: parallelize reduction) */
        #pragma omp single
        {
            rmsnorm(hidden, (const float *)ln1_weight, ln1_out, embed_dim, eps);
            quantize_row_q8_k(ln1_out, ln1_q8, aligned_embed);
        }
        /* Implicit barrier after single */

        /* Step 2: QKV projections (parallel) */
        gemv_q4_k_q8_k_parallel_simd(q_vec, WQ, ln1_q8, H * head_dim, aligned_embed, ith, nth);
        gemv_q4_k_q8_k_parallel_simd(k_vec, WK, ln1_q8, H_kv * head_dim, aligned_embed, ith, nth);
        gemv_q4_k_q8_k_parallel_simd(v_vec, WV, ln1_q8, H_kv * head_dim, aligned_embed, ith, nth);

        #pragma omp barrier

        /* Step 3: RoPE, attention, etc. would go here (single-threaded for now) */
        #pragma omp single
        {
            /* Copy K/V to cache */
            const int kv_head_stride = max_seq * aligned_head;
            for (int h = 0; h < H_kv; h++) {
                memcpy(k_cache + h * kv_head_stride + token_index * aligned_head,
                       k_vec + h * head_dim, head_dim * sizeof(float));
                memcpy(v_cache + h * kv_head_stride + token_index * aligned_head,
                       v_vec + h * head_dim, head_dim * sizeof(float));
            }

            /* TODO: RoPE, attention decode, etc. */
            /* For now, just copy q_vec to attn_out as placeholder */
            memcpy(attn_out, q_vec, H * head_dim * sizeof(float));
        }

        #pragma omp barrier

        /* Step 4: Output projection (parallel) */
        /* First quantize attention output */
        #pragma omp single
        {
            quantize_row_q8_k(attn_out, ln1_q8, H * aligned_head);  /* Reuse ln1_q8 */
        }

        gemv_q4_k_q8_k_parallel_simd(o_out, WO, ln1_q8, aligned_embed, H * head_dim, ith, nth);

        #pragma omp barrier

        /* Step 5: Residual add (parallel) */
        residual_add_parallel(hidden, o_out, hidden, embed_dim, ith, nth);

        #pragma omp barrier

        /* ============================================================
         * MLP BLOCK
         * ============================================================ */

        /* Step 6: RMSNorm */
        #pragma omp single
        {
            rmsnorm(hidden, (const float *)ln2_weight, ln2_out, embed_dim, eps);
            quantize_row_q8_k(ln2_out, ln2_q8, aligned_embed);
        }

        #pragma omp barrier

        /* Step 7: Gate and Up projections (parallel) */
        gemv_q4_k_q8_k_parallel_simd(gate_buf, W_gate, ln2_q8, aligned_inter, aligned_embed, ith, nth);
        gemv_q4_k_q8_k_parallel_simd(up_buf, W_up, ln2_q8, aligned_inter, aligned_embed, ith, nth);

        #pragma omp barrier

        /* Step 8: SwiGLU (parallel element-wise) */
        const int dr = (intermediate + nth - 1) / nth;
        const int r0 = dr * ith;
        const int r1 = (r0 + dr < intermediate) ? (r0 + dr) : intermediate;

        for (int i = r0; i < r1; i++) {
            float g = gate_buf[i];
            float silu_g = g / (1.0f + expf(-g));
            swiglu_buf[i] = silu_g * up_buf[i];
        }

        #pragma omp barrier

        /* Step 9: Down projection */
        #pragma omp single
        {
            quantize_row_q8_k(swiglu_buf, down_q8, aligned_inter);
        }

        gemv_q4_k_q8_k_parallel_simd(mlp_out, W_down, down_q8, aligned_embed, aligned_inter, ith, nth);

        #pragma omp barrier

        /* Step 10: Final residual add */
        residual_add_parallel(hidden, mlp_out, hidden, embed_dim, ith, nth);
    }
}

/* ============================================================================
 * CONFIGURATION
 * ============================================================================ */

/* Optimal thread count for decode on memory-bound systems */
int get_optimal_decode_threads(void)
{
    int max_threads = omp_get_max_threads();

    /* For memory-bound workloads, 4 threads is often optimal.
     * More threads hit memory bandwidth limits with diminishing returns.
     * See MEMORY_BANDWIDTH_ANALYSIS.md for details. */
    if (max_threads >= 4) {
        return 4;
    }
    return max_threads;
}

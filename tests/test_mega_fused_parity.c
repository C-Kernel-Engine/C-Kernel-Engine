/**
 * @file test_mega_fused_parity.c
 * @brief Test numerical parity between mega-fused and separate kernels
 *
 * Build: make test-mega-fused-parity
 * Run: ./build/test-mega-fused-parity
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>

#include "ckernel_quant.h"
#include "../src/kernels/fused/mega_fused_attention_decode_q5_0.h"

/* External kernels for comparison */
extern void rmsnorm_forward(const float *input, const float *gamma, float *output,
                            float *rstd, int T, int D, int AD, float eps);
extern void attention_forward_decode_head_major_gqa_flash(
    const float *q_token, const float *k_cache, const float *v_cache,
    float *out_token, int num_heads, int num_kv_heads, int kv_tokens,
    int cache_capacity, int head_dim, int aligned_head_dim);
extern void quantize_row_q8_0(const float *x, void *vy, int k);
extern void quantize_row_q5_0(const float *x, void *vy, int k);
extern void vec_dot_q5_0_q8_0(int n, float *s, const void *vx, const void *vy);
extern void vec_dot_q8_0_q8_0(int n, float *s, const void *vx, const void *vy);

/* ============================================================================
 * TEST INFRASTRUCTURE
 * ============================================================================ */

static double get_time_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

static float random_float(void) {
    return (float)rand() / RAND_MAX * 2.0f - 1.0f;
}

static float max_diff(const float *a, const float *b, int n) {
    float max_d = 0.0f;
    for (int i = 0; i < n; i++) {
        float d = fabsf(a[i] - b[i]);
        if (d > max_d) max_d = d;
    }
    return max_d;
}

/* ============================================================================
 * SEPARATE KERNEL IMPLEMENTATION (reference)
 * ============================================================================ */

static void reference_attention_decode(
    float *output,
    const float *input,
    const float *residual,
    const void *wq_q5_0,
    const void *wk_q5_0,
    const void *wv_q8_0,
    const void *wo_q5_0,
    const float *ln_gamma,
    const float *bq,
    const float *bk,
    const float *bv,
    const float *bo,
    float *kv_cache_k,
    float *kv_cache_v,
    const float *rope_cos,
    const float *rope_sin,
    int pos,
    int AE,
    int H,
    int KV,
    int AD,
    int cache_capacity,
    float eps)
{
    /* Allocate intermediate buffers */
    float *rmsnorm_out = malloc(AE * sizeof(float));
    float *rstd = malloc(AE * sizeof(float));
    float *q = malloc(H * AD * sizeof(float));
    float *k = malloc(KV * AD * sizeof(float));
    float *v = malloc(KV * AD * sizeof(float));
    float *attn_out = malloc(H * AD * sizeof(float));

    int q8_blocks = (AE + QK8_0 - 1) / QK8_0;
    block_q8_0 *x_q8 = malloc(q8_blocks * sizeof(block_q8_0));

    /* Step 1: RMSNorm */
    rmsnorm_forward(input, ln_gamma, rmsnorm_out, rstd, 1, AE, AD, eps);

    /* Step 2-4: Q, K, V projections */
    {
        const block_q5_0 *wq = (const block_q5_0 *)wq_q5_0;
        const block_q5_0 *wk = (const block_q5_0 *)wk_q5_0;
        const block_q8_0 *wv = (const block_q8_0 *)wv_q8_0;
        int blocks_per_row = AE / QK5_0;

        quantize_row_q8_0(rmsnorm_out, x_q8, AE);

        /* Q projection */
        for (int row = 0; row < H * AD; row++) {
            float dot;
            vec_dot_q5_0_q8_0(AE, &dot, &wq[row * blocks_per_row], x_q8);
            q[row] = dot + (bq ? bq[row] : 0.0f);
        }

        /* K projection */
        for (int row = 0; row < KV * AD; row++) {
            float dot;
            vec_dot_q5_0_q8_0(AE, &dot, &wk[row * blocks_per_row], x_q8);
            k[row] = dot + (bk ? bk[row] : 0.0f);
        }

        /* V projection */
        int blocks_per_row_q8 = AE / QK8_0;
        for (int row = 0; row < KV * AD; row++) {
            float dot;
            vec_dot_q8_0_q8_0(AE, &dot, &wv[row * blocks_per_row_q8], x_q8);
            v[row] = dot + (bv ? bv[row] : 0.0f);
        }
    }

    /* Step 5: RoPE */
    {
        const int D = AD / 2;
        const float *cos_row = &rope_cos[pos * D];
        const float *sin_row = &rope_sin[pos * D];

        for (int h = 0; h < H; h++) {
            float *q_head = &q[h * AD];
            for (int d = 0; d < D; d++) {
                float q0 = q_head[d];
                float q1 = q_head[d + D];
                q_head[d] = q0 * cos_row[d] - q1 * sin_row[d];
                q_head[d + D] = q0 * sin_row[d] + q1 * cos_row[d];
            }
        }

        for (int kv_idx = 0; kv_idx < KV; kv_idx++) {
            float *k_head = &k[kv_idx * AD];
            for (int d = 0; d < D; d++) {
                float k0 = k_head[d];
                float k1 = k_head[d + D];
                k_head[d] = k0 * cos_row[d] - k1 * sin_row[d];
                k_head[d + D] = k0 * sin_row[d] + k1 * cos_row[d];
            }
        }
    }

    /* Step 6: KV cache store */
    {
        const size_t kv_stride = (size_t)cache_capacity * AD;
        for (int kv_idx = 0; kv_idx < KV; kv_idx++) {
            float *k_cache = &kv_cache_k[kv_idx * kv_stride];
            float *v_cache = &kv_cache_v[kv_idx * kv_stride];
            const int offset = pos * AD;
            for (int d = 0; d < AD; d++) {
                k_cache[offset + d] = k[kv_idx * AD + d];
                v_cache[offset + d] = v[kv_idx * AD + d];
            }
        }
    }

    /* Step 7: Flash attention decode */
    attention_forward_decode_head_major_gqa_flash(
        q, kv_cache_k, kv_cache_v,
        attn_out, H, KV, pos + 1, cache_capacity, AD, AD);

    /* Step 8: O projection + residual */
    {
        const block_q5_0 *wo = (const block_q5_0 *)wo_q5_0;
        int blocks_per_row = (H * AD) / QK5_0;

        int q8_blocks_attn = (H * AD + QK8_0 - 1) / QK8_0;
        block_q8_0 *attn_q8 = malloc(q8_blocks_attn * sizeof(block_q8_0));
        quantize_row_q8_0(attn_out, attn_q8, H * AD);

        for (int e = 0; e < AE; e++) {
            float dot;
            vec_dot_q5_0_q8_0(H * AD, &dot, &wo[e * blocks_per_row], attn_q8);
            output[e] = dot + (bo ? bo[e] : 0.0f) + residual[e];
        }

        free(attn_q8);
    }

    free(rmsnorm_out);
    free(rstd);
    free(q);
    free(k);
    free(v);
    free(attn_out);
    free(x_q8);
}

/* ============================================================================
 * MAIN TEST
 * ============================================================================ */

int main(int argc, char *argv[]) {
    (void)argc; (void)argv;

    printf("=== Mega-Fused Attention Decode Parity Test ===\n\n");

    /* Test configuration - Qwen 0.5B-like dimensions */
    const int AE = 896;       /* Aligned embed dim */
    const int H = 14;         /* Num heads */
    const int KV = 2;         /* Num KV heads */
    const int AD = 64;        /* Head dim */
    const int cache_capacity = 128;  /* Smaller for test */
    const float eps = 1e-5f;
    const int pos = 5;        /* Test position */

    printf("Config: AE=%d, H=%d, KV=%d, AD=%d, pos=%d\n", AE, H, KV, AD, pos);

    /* Seed random */
    srand(42);

    /* Allocate buffers */
    float *input = malloc(AE * sizeof(float));
    float *residual = malloc(AE * sizeof(float));
    float *ln_gamma = malloc(AE * sizeof(float));
    float *bq = malloc(H * AD * sizeof(float));
    float *bk = malloc(KV * AD * sizeof(float));
    float *bv = malloc(KV * AD * sizeof(float));
    float *bo = malloc(AE * sizeof(float));
    float *output_ref = malloc(AE * sizeof(float));
    float *output_fused = malloc(AE * sizeof(float));

    /* Allocate weights (Q5_0 and Q8_0) */
    int wq_blocks = (H * AD) * (AE / QK5_0);
    int wk_blocks = (KV * AD) * (AE / QK5_0);
    int wv_blocks = (KV * AD) * (AE / QK8_0);
    int wo_blocks = AE * ((H * AD) / QK5_0);

    block_q5_0 *wq = malloc(wq_blocks * sizeof(block_q5_0));
    block_q5_0 *wk = malloc(wk_blocks * sizeof(block_q5_0));
    block_q8_0 *wv = malloc(wv_blocks * sizeof(block_q8_0));
    block_q5_0 *wo = malloc(wo_blocks * sizeof(block_q5_0));

    /* Allocate KV cache (two copies for separate tests) */
    size_t kv_cache_size = KV * cache_capacity * AD * sizeof(float);
    float *kv_cache_k_ref = malloc(kv_cache_size);
    float *kv_cache_v_ref = malloc(kv_cache_size);
    float *kv_cache_k_fused = malloc(kv_cache_size);
    float *kv_cache_v_fused = malloc(kv_cache_size);

    /* RoPE tables */
    int D = AD / 2;
    float *rope_cos = malloc(cache_capacity * D * sizeof(float));
    float *rope_sin = malloc(cache_capacity * D * sizeof(float));

    /* Scratch buffer for fused kernel */
    int scratch_size = mega_fused_attention_decode_scratch_size(AE, H, KV, AD);
    void *scratch = malloc(scratch_size);

    printf("Scratch size: %d bytes\n", scratch_size);

    /* Initialize random data */
    for (int i = 0; i < AE; i++) {
        input[i] = random_float() * 0.1f;
        residual[i] = random_float() * 0.1f;
        ln_gamma[i] = 1.0f + random_float() * 0.01f;
        bo[i] = random_float() * 0.01f;
    }

    for (int i = 0; i < H * AD; i++) bq[i] = random_float() * 0.01f;
    for (int i = 0; i < KV * AD; i++) {
        bk[i] = random_float() * 0.01f;
        bv[i] = random_float() * 0.01f;
    }

    /* Initialize quantized weights from random FP32 */
    {
        float *tmp = malloc(AE * sizeof(float));
        for (int row = 0; row < H * AD; row++) {
            for (int i = 0; i < AE; i++) tmp[i] = random_float() * 0.01f;
            quantize_row_q5_0(tmp, &wq[row * (AE / QK5_0)], AE);
        }
        for (int row = 0; row < KV * AD; row++) {
            for (int i = 0; i < AE; i++) tmp[i] = random_float() * 0.01f;
            quantize_row_q5_0(tmp, &wk[row * (AE / QK5_0)], AE);
        }
        for (int row = 0; row < KV * AD; row++) {
            for (int i = 0; i < AE; i++) tmp[i] = random_float() * 0.01f;
            quantize_row_q8_0(tmp, &wv[row * (AE / QK8_0)], AE);
        }

        float *tmp2 = malloc(H * AD * sizeof(float));
        for (int row = 0; row < AE; row++) {
            for (int i = 0; i < H * AD; i++) tmp2[i] = random_float() * 0.01f;
            quantize_row_q5_0(tmp2, &wo[row * ((H * AD) / QK5_0)], H * AD);
        }
        free(tmp);
        free(tmp2);
    }

    /* Initialize RoPE */
    for (int t = 0; t < cache_capacity; t++) {
        for (int d = 0; d < D; d++) {
            float theta = (float)t * powf(10000.0f, -2.0f * d / AD);
            rope_cos[t * D + d] = cosf(theta);
            rope_sin[t * D + d] = sinf(theta);
        }
    }

    /* Initialize KV caches with some prior context */
    memset(kv_cache_k_ref, 0, kv_cache_size);
    memset(kv_cache_v_ref, 0, kv_cache_size);
    memcpy(kv_cache_k_fused, kv_cache_k_ref, kv_cache_size);
    memcpy(kv_cache_v_fused, kv_cache_v_ref, kv_cache_size);

    /* Pre-fill some cache entries */
    for (int p = 0; p < pos; p++) {
        for (int kv_idx = 0; kv_idx < KV; kv_idx++) {
            for (int d = 0; d < AD; d++) {
                size_t idx = kv_idx * cache_capacity * AD + p * AD + d;
                float val = random_float() * 0.1f;
                kv_cache_k_ref[idx] = val;
                kv_cache_v_ref[idx] = random_float() * 0.1f;
                kv_cache_k_fused[idx] = kv_cache_k_ref[idx];
                kv_cache_v_fused[idx] = kv_cache_v_ref[idx];
            }
        }
    }

    printf("\n--- Running Reference (Separate Kernels) ---\n");
    double t0 = get_time_ms();
    reference_attention_decode(
        output_ref, input, residual,
        wq, wk, wv, wo, ln_gamma,
        bq, bk, bv, bo,
        kv_cache_k_ref, kv_cache_v_ref,
        rope_cos, rope_sin,
        pos, AE, H, KV, AD, cache_capacity, eps);
    double t1 = get_time_ms();
    printf("Reference time: %.3f ms\n", t1 - t0);

    printf("\n--- Running Mega-Fused Kernel ---\n");
    t0 = get_time_ms();
    mega_fused_attention_decode_q5_0(
        output_fused, input, residual,
        wq, wk, wv, wo, ln_gamma,
        bq, bk, bv, bo,
        kv_cache_k_fused, kv_cache_v_fused,
        rope_cos, rope_sin,
        pos, AE, AE, H, KV, AD, AD, cache_capacity, eps, scratch);
    t1 = get_time_ms();
    printf("Fused time: %.3f ms\n", t1 - t0);

    /* Check numerical parity */
    printf("\n--- Numerical Parity Check ---\n");
    float max_output_diff = max_diff(output_ref, output_fused, AE);
    float max_k_cache_diff = max_diff(kv_cache_k_ref, kv_cache_k_fused, KV * cache_capacity * AD);
    float max_v_cache_diff = max_diff(kv_cache_v_ref, kv_cache_v_fused, KV * cache_capacity * AD);

    printf("Max output diff: %.6e\n", max_output_diff);
    printf("Max K cache diff: %.6e\n", max_k_cache_diff);
    printf("Max V cache diff: %.6e\n", max_v_cache_diff);

    const float tolerance = 1e-4f;
    int pass = (max_output_diff < tolerance) &&
               (max_k_cache_diff < tolerance) &&
               (max_v_cache_diff < tolerance);

    printf("\n%s (tolerance: %.0e)\n",
           pass ? "PASS ✓" : "FAIL ✗", tolerance);

    /* Benchmark multiple iterations */
    printf("\n--- Benchmark (100 iterations) ---\n");
    const int iters = 100;

    t0 = get_time_ms();
    for (int i = 0; i < iters; i++) {
        reference_attention_decode(
            output_ref, input, residual,
            wq, wk, wv, wo, ln_gamma,
            bq, bk, bv, bo,
            kv_cache_k_ref, kv_cache_v_ref,
            rope_cos, rope_sin,
            pos, AE, H, KV, AD, cache_capacity, eps);
    }
    t1 = get_time_ms();
    double ref_ms = (t1 - t0) / iters;

    t0 = get_time_ms();
    for (int i = 0; i < iters; i++) {
        mega_fused_attention_decode_q5_0(
            output_fused, input, residual,
            wq, wk, wv, wo, ln_gamma,
            bq, bk, bv, bo,
            kv_cache_k_fused, kv_cache_v_fused,
            rope_cos, rope_sin,
            pos, AE, AE, H, KV, AD, AD, cache_capacity, eps, scratch);
    }
    t1 = get_time_ms();
    double fused_ms = (t1 - t0) / iters;

    printf("Reference: %.3f ms/iter\n", ref_ms);
    printf("Fused:     %.3f ms/iter\n", fused_ms);
    printf("Speedup:   %.2fx\n", ref_ms / fused_ms);

    /* Cleanup */
    free(input);
    free(residual);
    free(ln_gamma);
    free(bq);
    free(bk);
    free(bv);
    free(bo);
    free(output_ref);
    free(output_fused);
    free(wq);
    free(wk);
    free(wv);
    free(wo);
    free(kv_cache_k_ref);
    free(kv_cache_v_ref);
    free(kv_cache_k_fused);
    free(kv_cache_v_fused);
    free(rope_cos);
    free(rope_sin);
    free(scratch);

    return pass ? 0 : 1;
}

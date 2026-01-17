/**
 * @file gemm_kernels_q5_0_sse_v2.c
 * @brief SSE-optimized GEMM kernels for Q5_0 x Q8_K quantization
 *
 * CK-ENGINE KERNEL RULES:
 * =======================
 * 1. NO malloc/free - memory via bump allocator, pointers passed in
 * 2. NO OpenMP - parallelization at orchestrator/codegen layer
 * 3. API must define: inputs, outputs, workspace, and memory layouts
 * 4. Pure computation - deterministic, no side effects
 *
 * After changes: make test && make llamacpp-parity-full
 */

#include <immintrin.h>
#include <stdint.h>
#include <string.h>
#include <stdio.h>

#include "ckernel_quant.h"

void quantize_row_q8_k(const float *x, void *vy, int k);
void gemm_nt_q5_0_ref(const float *A, const void *B, const float *bias, float *C, int M, int N, int K);

static inline float dot_q5_0_q8_k_32_sse(const block_q5_0 *bw, const block_q8_K *ba, int q8_offset) {
    const uint8_t *qs_w = bw->qs;
    const int8_t  *qs_a = ba->qs + q8_offset;
    
    uint32_t qh;
    memcpy(&qh, bw->qh, sizeof(qh));

    // Vectorize bit extraction
    // Load 16 bytes of low nibbles
    __m128i qs_vec = _mm_loadu_si128((const __m128i *)qs_w);
    __m128i mask_0f = _mm_set1_epi8(0x0F);
    
    __m128i w_lo = _mm_and_si128(qs_vec, mask_0f);
    __m128i w_hi = _mm_and_si128(_mm_srli_epi16(qs_vec, 4), mask_0f);

    // Load high bits from qh
    // This is still a bit scalar but we can use shuffle for some of it if needed
    // For now, let's just make sure we handle the -16 offset correctly in SIMD.
    
    uint8_t w[32];
    for (int j = 0; j < 16; j++) {
        w[j]    = (qs_w[j] & 0x0F) | (((qh >> (j + 0)) << 4) & 0x10);
        w[j+16] = (qs_w[j] >> 4)   | ((qh >> (j + 12)) & 0x10);
    }

    __m128i vw0 = _mm_loadu_si128((const __m128i *)&w[0]);
    __m128i vw1 = _mm_loadu_si128((const __m128i *)&w[16]);
    __m128i va0 = _mm_loadu_si128((const __m128i *)&qs_a[0]);
    __m128i va1 = _mm_loadu_si128((const __m128i *)&qs_a[16]);

    // Dot product: unsigned 8-bit * signed 8-bit -> signed 16-bit
    __m128i p0 = _mm_maddubs_epi16(vw0, va0);
    __m128i p1 = _mm_maddubs_epi16(vw1, va1);

    // Sum to i32
    __m128i one = _mm_set1_epi16(1);
    __m128i s0 = _mm_madd_epi16(p0, one);
    __m128i s1 = _mm_madd_epi16(p1, one);
    __m128i acc_i32 = _mm_add_epi32(s0, s1);

    // Horizontal sum of i32
    acc_i32 = _mm_add_epi32(acc_i32, _mm_shuffle_epi32(acc_i32, _MM_SHUFFLE(1, 0, 3, 2)));
    acc_i32 = _mm_add_epi32(acc_i32, _mm_shuffle_epi32(acc_i32, _MM_SHUFFLE(0, 1, 0, 1)));
    int32_t dot_wa = _mm_cvtsi128_si32(acc_i32);

    // sum((w - 16) * a) = sum(w*a) - 16 * sum(a)
    int32_t sum_a = (int32_t)ba->bsums[q8_offset/16] + (int32_t)ba->bsums[q8_offset/16 + 1];
    
    float result = ((float)dot_wa - 16.0f * (float)sum_a) * CK_FP16_TO_FP32(bw->d) * ba->d;
    return result;
}

void gemm_nt_q5_0_sse_v2(const float *A,
                         const void *B,
                         const float *bias,
                         float *C,
                         int M, int N, int K)
{
    if (K % QK_K != 0) {
        gemm_nt_q5_0_ref(A, B, bias, C, M, N, K);
        return;
    }

    size_t q8_size = (K / QK_K) * sizeof(block_q8_K);
    block_q8_K *A_q8 = (block_q8_K *)alloca(q8_size);

    const block_q5_0 *weights = (const block_q5_0 *)B;
    const int blocks_per_row = K / 32;

    for (int m = 0; m < M; m++) {
        quantize_row_q8_k(&A[m * K], A_q8, K);

        for (int n = 0; n < N; n++) {
            float sumf = 0.0f;
            const block_q5_0 *w_row = weights + n * blocks_per_row;

            for (int b = 0; b < blocks_per_row; b++) {
                int q8_block_idx = (b * 32) / QK_K;
                int q8_offset = (b * 32) % QK_K;
                sumf += dot_q5_0_q8_k_32_sse(&w_row[b], &A_q8[q8_block_idx], q8_offset);
            }

            C[m * N + n] = sumf + (bias ? bias[n] : 0.0f);
        }
    }
}

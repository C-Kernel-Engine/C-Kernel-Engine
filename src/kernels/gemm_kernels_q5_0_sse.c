#include <immintrin.h>
#include <stdint.h>
#include <string.h>

#include "ckernel_quant.h"

// SSE4.1 implementation of GEMM_NT Q5_0 (Weights Q5_0, Activations FP32)
// Compatible with AVX/SSE4.1 CPUs

void gemm_nt_q5_0_sse(const float *A,
                      const void *B,
                      const float *bias,
                      float *C,
                      int M, int N, int K)
{
    const block_q5_0 *blocks = (const block_q5_0 *)B;
    const int blocks_per_row = K / QK5_0;

    const __m128i mask_lo = _mm_set1_epi8(0x0F);
    const __m128i sixteen = _mm_set1_epi8(16);

    for (int m = 0; m < M; m++) {
        const float *a_row = &A[m * K];

        for (int n = 0; n < N; n++) {
            __m128 sum_v = _mm_setzero_ps();
            
            for (int b = 0; b < blocks_per_row; b++) {
                const block_q5_0 *block = &blocks[n * blocks_per_row + b];
                float d_val = CK_FP16_TO_FP32(block->d);
                __m128 d = _mm_set1_ps(d_val);
                const float *ap = &a_row[b * QK5_0];

                uint32_t qh_val;
                memcpy(&qh_val, block->qh, sizeof(qh_val));

                // Load 16 bytes (32 weights compressed)
                __m128i qs = _mm_loadu_si128((const __m128i *)block->qs);
                
                // Low nibbles (0-15)
                __m128i lo = _mm_and_si128(qs, mask_lo);
                // High nibbles (16-31) - shift right by 4
                __m128i hi = _mm_and_si128(_mm_srli_epi16(qs, 4), mask_lo);

                // Now we need to add the high bits from qh
                // qh has 32 bits.
                // For j=0..15: bit j of qh.
                // For j=16..31: bit j-16+12 = j-4 of qh?
                // Wait, ref code:
                // j=0..15: qh >> j
                // j=16..31: qh >> (j-16 + 12) = qh >> (j-4)
                
                // We will process in chunks of 16 weights (128-bit)? 
                // No, converting int8 to float takes space.
                // We need to unpack to 32-bit integers to convert to float.
                
                // Strategy: Extract 32 bytes of weights.
                uint8_t q_vals[32];
                // Vectorized extraction is painful in SSE without AVX2 gathers.
                // Scalar unpacking of 32 bytes is better than bit-level math inside the loop.
                // Actually, let's use the SIMD for the float math (the heavy part) 
                // and scalar for the unpacking if needed, OR try to vectorize unpacking.

                // Vectorized unpacking:
                // lo (16 bytes). We need to add (qh & (1<<j)) << 4.
                // Construct a 16-byte mask from qh bits 0-15.
                // Then another mask from qh bits 12-27.
                
                // Optimized bit extraction is hard in SSE. 
                // Let's do a hybrid: unpack to stack buffer, then load as floats.
                // Or just do scalar unpack since loop count is small (32).
                // But the main cost is the 32 muls.
                
                for (int j = 0; j < 16; j++) {
                     uint8_t v = (block->qs[j] & 0x0F) | (((qh_val >> j) & 1) << 4);
                     q_vals[j] = v;
                }
                for (int j = 0; j < 16; j++) {
                     uint8_t v = (block->qs[j] >> 4) | (((qh_val >> (j+12)) & 1) << 4);
                     q_vals[j+16] = v;
                }

                // Now we have 32 uint8_t values 0..31.
                // Subtract 16, convert to float, mul by d, mul by x.
                
                // Process 32 elements in 8x __m128 ops
                for (int k=0; k<32; k+=4) {
                    // Load 4 bytes
                    // Convert to 4 floats
                    // (x - 16) * d * a
                    
                    float w0 = (float)((int)q_vals[k] - 16) * d_val;
                    float w1 = (float)((int)q_vals[k+1] - 16) * d_val;
                    float w2 = (float)((int)q_vals[k+2] - 16) * d_val;
                    float w3 = (float)((int)q_vals[k+3] - 16) * d_val;
                    
                    __m128 w = _mm_set_ps(w3, w2, w1, w0);
                    __m128 x = _mm_loadu_ps(&ap[k]);
                    sum_v = _mm_add_ps(sum_v, _mm_mul_ps(w, x));
                }
            }

            // Hsum
            float output;
            _mm_store_ss(&output, _mm_hadd_ps(_mm_hadd_ps(sum_v, sum_v), sum_v));
            C[m * N + n] = output + (bias ? bias[n] : 0.0f);
        }
    }
}

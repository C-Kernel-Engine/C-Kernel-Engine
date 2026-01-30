/**
 * @file mega_fused_attention_decode_q5_0.h
 * @brief Mega-fused attention decode with Q5_0 weights - Header
 *
 * This header declares the mega-fused attention decode kernel that combines
 * 9 separate operations into a single fused kernel call:
 *   1. RMSNorm
 *   2. Q projection (Q5_0) with bias
 *   3. K projection (Q5_0) with bias
 *   4. V projection (Q8_0) with bias
 *   5. RoPE application
 *   6. KV cache store
 *   7. Flash attention decode (GQA-aware)
 *   8. O projection (Q5_0) with bias
 *   9. Residual add
 */

#ifndef MEGA_FUSED_ATTENTION_DECODE_Q5_0_H
#define MEGA_FUSED_ATTENTION_DECODE_Q5_0_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Calculate scratch buffer size needed for the kernel
 *
 * @param AE Aligned embedding dimension (multiple of 64)
 * @param H Number of query heads
 * @param KV Number of key/value heads
 * @param AD Head dimension
 * @return Size in bytes needed for scratch buffer
 */
int mega_fused_attention_decode_scratch_size(int AE, int H, int KV, int AD);

/**
 * @brief Serial mega-fused attention decode kernel
 *
 * @param output Output [AE] (final result, after residual add)
 * @param input Input activation [AE]
 * @param residual Residual input for add [AE]
 * @param wq_q5_0 Q projection weights [H*AD, AE] Q5_0
 * @param wk_q5_0 K projection weights [KV*AD, AE] Q5_0
 * @param wv_q8_0 V projection weights [KV*AD, AE] Q8_0
 * @param wo_q5_0 O projection weights [AE, H*AD] Q5_0
 * @param ln_gamma RMSNorm gamma [AE]
 * @param bq Q bias [H*AD] or NULL
 * @param bk K bias [KV*AD] or NULL
 * @param bv V bias [KV*AD] or NULL
 * @param bo O bias [AE] or NULL
 * @param kv_cache_k K cache [KV, max_T, AD]
 * @param kv_cache_v V cache [KV, max_T, AD]
 * @param rope_cos RoPE cos [max_T, D]
 * @param rope_sin RoPE sin [max_T, D]
 * @param pos Current position (0-indexed)
 * @param embed_dim Original embedding dimension E
 * @param aligned_embed_dim Aligned embedding dimension AE
 * @param num_heads Number of query heads H
 * @param num_kv_heads Number of key/value heads KV
 * @param head_dim Head dimension AD
 * @param aligned_head_dim Aligned head dimension AAD
 * @param cache_capacity Maximum cache capacity max_T
 * @param eps RMSNorm epsilon
 * @param scratch Scratch buffer (>= scratch_size bytes)
 */
void mega_fused_attention_decode_q5_0(
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
    int embed_dim,
    int aligned_embed_dim,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int aligned_head_dim,
    int cache_capacity,
    float eps,
    void *scratch);

/**
 * @brief Parallel SIMD mega-fused attention decode kernel (threadpool-aware)
 *
 * Parallelizes across attention heads using (ith, nth) pattern.
 * Each thread processes a subset of heads.
 *
 * IMPORTANT: Caller must ensure barrier sync between phases:
 *   Phase 1 (ith==0 only): RMSNorm, Q/K/V projection, RoPE, KV cache store
 *   -- BARRIER --
 *   Phase 2 (all threads): Attention for assigned heads
 *   -- BARRIER --
 *   Phase 3 (ith==0 only): O projection and residual add
 *
 * @param ith Thread index (0 to nth-1)
 * @param nth Total number of threads
 * (other parameters same as serial version)
 */
void mega_fused_attention_decode_q5_0_parallel_simd(
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
    int embed_dim,
    int aligned_embed_dim,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int aligned_head_dim,
    int cache_capacity,
    float eps,
    void *scratch,
    int ith,
    int nth);

#ifdef __cplusplus
}
#endif

#endif /* MEGA_FUSED_ATTENTION_DECODE_Q5_0_H */

/**
 * @file mega_fused_attention.h
 * @brief Mega-Fused Attention Kernel
 *
 * Holy grail fusion: RMSNorm → QKV → RoPE → Flash Attention → OutProj + Residual
 *
 * All intermediates stay in registers/L1/L2. Single DRAM round-trip.
 *
 * Memory Reduction:
 *   Before: ~32KB intermediates per layer (stack/heap)
 *   After:  ~8KB total (input + output only)
 *   Reduction: 4-5× per layer, ~100× for full model
 *
 * Performance Target:
 *   Move from memory-bound to compute-bound
 *   Expected speedup: 5-10× for attention-heavy workloads
 */

#ifndef MEGA_FUSED_ATTENTION_H
#define MEGA_FUSED_ATTENTION_H

#include <stdint.h>

#include "ckernel_dtype.h"

/*============================================================================
 * Configuration
 *============================================================================*/

/* Tile sizes for streaming through cache hierarchy */
#ifndef MEGA_FUSE_Q_TILE
#define MEGA_FUSE_Q_TILE 64
#endif

#ifndef MEGA_FUSE_KV_TILE
#define MEGA_FUSE_KV_TILE 64
#endif

/*============================================================================
 * Mega-Fused Attention API
 *============================================================================*/

/**
 * @brief Mega-fused attention for decode mode (single token)
 *
 * This is the "holy grail" - all operations fused, no intermediates to DRAM.
 *
 * @param output           Output [aligned_embed_dim] (includes residual add)
 * @param input            Input [aligned_embed_dim]
 * @param residual         Residual input [aligned_embed_dim] (or NULL)
 * @param ln1_gamma        RMSNorm gamma [embed_dim]
 * @param wq               Q weights (quantized) [num_heads * aligned_head_dim * aligned_embed_dim]
 * @param bq               Q bias [num_heads * aligned_head_dim] (or NULL)
 * @param wq_dt            Q weight dtype (CK_DT_Q5_0/CK_DT_Q8_0/CK_DT_FP32)
 * @param wk               K weights (quantized) [num_kv_heads * aligned_head_dim * aligned_embed_dim]
 * @param bk               K bias [num_kv_heads * aligned_head_dim] (or NULL)
 * @param wk_dt            K weight dtype (CK_DT_Q5_0/CK_DT_Q8_0/CK_DT_FP32)
 * @param wv               V weights (quantized) [num_kv_heads * aligned_head_dim * aligned_embed_dim]
 * @param bv               V bias [num_kv_heads * aligned_head_dim] (or NULL)
 * @param wv_dt            V weight dtype (CK_DT_Q5_0/CK_DT_Q8_0/CK_DT_FP32)
 * @param wo               Output projection weights (quantized) [aligned_embed_dim * aligned_embed_dim]
 * @param bo               Output bias [aligned_embed_dim] (or NULL)
 * @param wo_dt            Output weight dtype (CK_DT_Q5_0/CK_DT_FP32)
 * @param kv_cache_k       KV cache for K [num_kv_heads * cache_capacity * aligned_head_dim]
 * @param kv_cache_v       KV cache for V [num_kv_heads * cache_capacity * aligned_head_dim]
 * @param rope_cos         RoPE cos [max_seq, head_dim/2]
 * @param rope_sin         RoPE sin [max_seq, head_dim/2]
 * @param pos              Current position in sequence
 * @param embed_dim        Model hidden dimension (unpadded)
 * @param aligned_embed_dim Aligned hidden dimension
 * @param num_heads        Number of attention heads
 * @param num_kv_heads     Number of KV heads (for GQA)
 * @param head_dim         Head dimension (unpadded)
 * @param aligned_head_dim Aligned head dimension
 * @param cache_capacity   KV cache capacity (stride in tokens)
 * @param eps              RMSNorm epsilon
 * @param scratch          Scratch buffer from mega_fused_attention_prefill_scratch_size()
 */
void mega_fused_attention_decode(
    float *output,
    const float *input,
    const float *residual,
    const float *ln1_gamma,
    const float *wq, const float *bq,
    const float *wk, const float *bk,
    const float *wv, const float *bv,
    const float *wo, const float *bo,
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
    float eps
);

/**
 * @brief Mega-fused attention for prefill mode (multiple tokens)
 *
 * @param output           Output [tokens, aligned_embed_dim] (includes residual add)
 * @param input            Input [tokens, aligned_embed_dim]
 * @param residual         Residual input [tokens, aligned_embed_dim] (or NULL)
 * @param ln1_gamma        RMSNorm gamma [embed_dim]
 * @param wq               Q weights [num_heads * aligned_head_dim * aligned_embed_dim]
 * @param bq               Q bias [num_heads * aligned_head_dim] (or NULL)
 * @param wk               K weights [num_kv_heads * aligned_head_dim * aligned_embed_dim]
 * @param bk               K bias [num_kv_heads * aligned_head_dim] (or NULL)
 * @param wv               V weights [num_kv_heads * aligned_head_dim * aligned_embed_dim]
 * @param bv               V bias [num_kv_heads * aligned_head_dim] (or NULL)
 * @param wo               Output projection weights [num_heads * aligned_embed_dim * aligned_head_dim]
 * @param bo               Output bias [aligned_embed_dim] (or NULL)
 * @param kv_cache_k       KV cache for K [num_kv_heads * cache_capacity * aligned_head_dim]
 * @param kv_cache_v       KV cache for V [num_kv_heads * cache_capacity * aligned_head_dim]
 * @param rope_cos         RoPE cos [max_seq, head_dim/2]
 * @param rope_sin         RoPE sin [max_seq, head_dim/2]
 * @param start_pos        Starting position in KV cache
 * @param tokens           Number of tokens to process
 * @param cache_capacity   KV cache capacity (stride in tokens)
 * @param embed_dim        Model hidden dimension (unpadded)
 * @param aligned_embed_dim Aligned hidden dimension
 * @param num_heads        Number of attention heads
 * @param num_kv_heads     Number of KV heads
 * @param head_dim         Head dimension (unpadded)
 * @param aligned_head_dim Aligned head dimension
 * @param eps              RMSNorm epsilon
 */
void mega_fused_attention_prefill(
    float *output,
    const float *input,
    const float *residual,
    const float *ln1_gamma,
    const void *wq, const float *bq, CKDataType wq_dt,
    const void *wk, const float *bk, CKDataType wk_dt,
    const void *wv, const float *bv, CKDataType wv_dt,
    const void *wo, const float *bo, CKDataType wo_dt,
    float *kv_cache_k,
    float *kv_cache_v,
    const float *rope_cos,
    const float *rope_sin,
    int start_pos,
    int tokens,
    int cache_capacity,
    int embed_dim,
    int aligned_embed_dim,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int aligned_head_dim,
    float eps,
    void *scratch
);

/** @brief Get scratch buffer size for mega_fused_attention_prefill */
size_t mega_fused_attention_prefill_scratch_size(int tokens,
                                                 int aligned_embed_dim,
                                                 int num_heads,
                                                 int aligned_head_dim);

/**
 * @brief Phase 1: Fused RMSNorm + QKV (intermediates in registers)
 *
 * Simpler step: Just fuse RMSNorm with QKV projection.
 * Q/K/V stay in stack buffers, not DRAM.
 */
void mega_fuse_rmsnorm_qkv(
    float *q_out,           // [num_heads * head_dim]
    float *k_out,           // [num_kv_heads * head_dim]
    float *v_out,           // [num_kv_heads * head_dim]
    const float *input,     // [hidden]
    const float *gamma,     // [hidden]
    const float *W_qkv,
    const float *b_qkv,
    int hidden,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    float eps
);

/**
 * @brief Phase 2: Fused RMSNorm + QKV + RoPE
 *
 * Q/K stay in output buffers, RoPE applied in-place.
 */
void mega_fuse_rmsnorm_qkv_rope(
    float *q_out,
    float *k_out,
    float *v_out,
    const float *input,
    const float *gamma,
    const float *W_qkv,
    const float *b_qkv,
    const float *rope_cos,
    const float *rope_sin,
    int pos,
    int hidden,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int max_seq,
    float eps
);

/**
 * @brief Get optimal tile sizes for current CPU
 */
void mega_fuse_get_optimal_tiles(
    int *q_tile,            // Output: Q tile size
    int *kv_tile,           // Output: KV tile size
    int head_dim
);

/**
 * @brief Report memory savings from mega-fusion
 */
void mega_fuse_report_stats(
    int hidden,
    int num_layers,
    int seq_len
);

#endif /* MEGA_FUSED_ATTENTION_H */

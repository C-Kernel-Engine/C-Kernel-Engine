/**
 * @file mega_fused_attention.h
 * @brief Mega-Fused Attention Kernel
 *
 * Holy grail fusion: RMSNorm → QKV → RoPE → Flash Attention → OutProj
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
 * @param output           Output [hidden] - single DRAM write
 * @param input            Input [hidden] - single DRAM read
 * @param residual         Residual input [hidden] - read for residual add
 * @param W_qkv            QKV weight matrix [3*hidden, hidden]
 * @param b_qkv            QKV bias [3*hidden] (or NULL)
 * @param W_o              Output projection weight [hidden, hidden]
 * @param b_o              Output bias [hidden] (or NULL)
 * @param kv_cache_k       KV cache for K [seq, hidden_kv]
 * @param kv_cache_v       KV cache for V [seq, hidden_kv]
 * @param rope_cos         RoPE cos [max_seq, head_dim/2]
 * @param rope_sin         RoPE sin [max_seq, head_dim/2]
 * @param pos              Current position in sequence
 * @param hidden           Model hidden dimension
 * @param num_heads        Number of attention heads
 * @param num_kv_heads     Number of KV heads (for GQA)
 * @param head_dim         Head dimension
 * @param max_seq          Maximum sequence length (for RoPE)
 * @param eps              RMSNorm epsilon
 */
void mega_fused_attention_decode(
    float *output,
    const float *input,
    const float *residual,
    const float *W_qkv,
    const float *b_qkv,
    const float *W_o,
    const float *b_o,
    float *kv_cache_k,
    float *kv_cache_v,
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
 * @brief Mega-fused attention for prefill mode (multiple tokens)
 *
 * @param output           Output [tokens, hidden]
 * @param input            Input [tokens, hidden]
 * @param residual         Residual input [tokens, hidden]
 * @param W_qkv            QKV weight matrix [3*hidden, hidden]
 * @param W_o              Output projection weight [hidden, hidden]
 * @param kv_cache_k       KV cache for K [seq, hidden_kv]
 * @param kv_cache_v       KV cache for V [seq, hidden_kv]
 * @param rope_cos         RoPE cos [max_seq, head_dim/2]
 * @param rope_sin         RoPE sin [max_seq, head_dim/2]
 * @param start_pos        Starting position in KV cache
 * @param tokens           Number of tokens to process
 * @param total_kv_len     Total KV cache length
 * @param hidden           Model hidden dimension
 * @param num_heads        Number of attention heads
 * @param num_kv_heads     Number of KV heads
 * @param head_dim         Head dimension
 * @param max_seq          Maximum sequence length
 * @param eps              RMSNorm epsilon
 */
void mega_fused_attention_prefill(
    float *output,
    const float *input,
    const float *residual,
    const float *W_qkv,
    const float *W_o,
    float *kv_cache_k,
    float *kv_cache_v,
    const float *rope_cos,
    const float *rope_sin,
    int start_pos,
    int tokens,
    int total_kv_len,
    int hidden,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int max_seq,
    float eps
);

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

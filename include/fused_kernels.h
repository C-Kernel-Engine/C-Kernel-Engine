/**
 * @file fused_kernels.h
 * @brief Fused Kernel API for Cache-Aware Attention Fusion
 *
 * Key design principles:
 * 1. Kernels take output buffer as parameter (no internal malloc)
 * 2. Buffers are designed to fit in L1/L2 cache
 * 3. Kernels can chain: rmsnorm → QKV → RoPE → Flash → OutProj
 * 4. Per-head parallelization with L1 cache constraints
 *
 * Cache hierarchy (typical Xeon):
 *   - Registers: 32 ZMM × 64B = 2KB (AVX-512)
 *   - L1: 48KB per core
 *   - L2: 1-2MB per core
 *   - L3: 1.5MB × cores (shared)
 *
 * Per-head working set must fit in L1:
 *   Q_h (128B) + K_tile (8KB) + V_tile (8KB) + O_h (256B) = ~16.8KB
 *
 * Reference: docs/site/assets/per_head_fusion_math.svg
 */

#ifndef FUSED_KERNELS_H
#define FUSED_KERNELS_H

#include <stdint.h>

/*============================================================================
 * Cache-Aware Configuration
 *============================================================================*/

/**
 * @brief Compute optimal KV tile size for flash attention
 *
 * Formula: T_kv ≤ (S_L1 × 0.8 - 2×d×B) / (2×d×B)
 *
 * Uses 80% of L1 to leave room for prefetch and OS.
 *
 * @param l1_size L1 cache size in bytes (typically 49152 for 48KB)
 * @param head_dim Head dimension
 * @param bytes_per_elem Element size (2 for FP16, 4 for FP32)
 * @return Optimal KV tile size (multiple of 64 for cache line alignment)
 */
int fused_kernels_compute_kv_tile(
    int l1_size,
    int head_dim,
    int bytes_per_elem
);

/*============================================================================
 * Fused RMSNorm
 *============================================================================*/

/**
 * @brief Fused RMSNorm - writes to pre-allocated buffer
 *
 * Takes input, applies RMSNorm, writes to output buffer.
 * Buffer should be in L1/L2 for best performance.
 *
 * @param input Input tensor [hidden]
 * @param gamma Gamma parameter [hidden]
 * @param beta Beta parameter [hidden] or NULL (RMSNorm doesn't use beta typically)
 * @param output Output buffer [hidden] - pre-allocated, caller owns
 * @param hidden Hidden dimension
 * @param eps Epsilon for numerical stability
 */
void fused_rmsnorm(
    const float *input,
    const float *gamma,
    const float *beta,
    float *output,
    int hidden,
    float eps
);

/**
 * @brief Fused RMSNorm with fused QKV projection
 *
 * Computes RMSNorm and immediately projects to Q, K, V.
 * No intermediate buffer - Q/K/V written directly to output buffers.
 *
 * @param input Input tensor [hidden]
 * @param gamma RMSNorm gamma [hidden]
 * @param W_qkv QKV weight matrix [3*hidden, hidden]
 * @param b_qkv QKV bias [3*hidden] or NULL
 * @param q_out Output buffer for Q [num_heads * head_dim]
 * @param k_out Output buffer for K [num_kv_heads * head_dim]
 * @param v_out Output buffer for V [num_kv_heads * head_dim]
 * @param hidden Hidden dimension
 * @param num_heads Number of attention heads
 * @param num_kv_heads Number of KV heads
 * @param head_dim Head dimension
 * @param eps RMSNorm epsilon
 */
void fused_rmsnorm_qkv(
    const float *input,
    const float *gamma,
    const float *W_qkv,
    const float *b_qkv,
    float *q_out,
    float *k_out,
    float *v_out,
    int hidden,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    float eps
);

/*============================================================================
 * Fused RoPE
 *============================================================================*/

/**
 * @brief Fused RoPE application (in-place on pre-allocated buffers)
 *
 * Applies RoPE rotation to Q and K in their buffers.
 * Buffers stay in cache - no extra memory allocation.
 *
 * @param q Q tensor [num_heads * head_dim] - modified in place
 * @param k K tensor [num_kv_heads * head_dim] - modified in place
 * @param rope_cos RoPE cos table [max_seq, head_dim/2]
 * @param rope_sin RoPE sin table [max_seq, head_dim/2]
 * @param pos Current position in sequence
 * @param num_heads Number of Q heads
 * @param num_kv_heads Number of K/V heads
 * @param head_dim Head dimension
 * @param max_seq Maximum sequence length
 */
void fused_rope_inplace(
    float *q,
    float *k,
    const float *rope_cos,
    const float *rope_sin,
    int pos,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int max_seq
);

/*============================================================================
 * Fused Flash Attention
 *============================================================================*/

/**
 * @brief Fused Flash Attention for single head
 *
 * Online softmax with streaming KV tiles.
 * O, m, l stay in registers throughout.
 *
 * @param o_out Output buffer [head_dim] - pre-allocated
 * @param q Q vector for this head [head_dim]
 * @param kv_cache_k KV cache K [seq_len * num_kv_heads * head_dim]
 * @param kv_cache_v KV cache V [seq_len * num_kv_heads * head_dim]
 * @param kv_head_idx Which KV head this Q head uses
 * @param seq_len Current sequence length
 * @param head_dim Head dimension
 * @param kv_tile_size Tile size for KV streaming
 */
void fused_flash_attention_head(
    float *o_out,
    const float *q,
    const float *kv_cache_k,
    const float *kv_cache_v,
    int kv_head_idx,
    int seq_len,
    int head_dim,
    int kv_tile_size
);

/**
 * @brief Fused Flash Attention for all heads (parallel dispatch)
 *
 * Dispatches per-head attention to parallel cores.
 * Each head's working set fits in L1.
 *
 * @param o_out Output buffer [num_heads * head_dim]
 * @param q_all Q tensor for all heads [num_heads * head_dim]
 * @param kv_cache_k KV cache K [seq_len * num_kv_heads * head_dim]
 * @param kv_cache_v KV cache V [seq_len * num_kv_heads * head_dim]
 * @param num_heads Number of attention heads
 * @param num_kv_heads Number of KV heads
 * @param head_dim Head dimension
 * @param seq_len Current sequence length
 * @param kv_tile_size Tile size for KV streaming
 */
void fused_flash_attention_all_heads(
    float *o_out,
    const float *q_all,
    const float *kv_cache_k,
    const float *kv_cache_v,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int seq_len,
    int kv_tile_size
);

/*============================================================================
 * Fused Output Projection
 *============================================================================*/

/**
 * @brief Fused output projection with residual add
 *
 * Computes O @ W_o + residual, all in registers/L1.
 * Final store includes residual add.
 *
 * @param output Output buffer [hidden] - final DRAM write
 * @param o_all Concatenated O from all heads [hidden]
 * @param W_o Output projection weights [hidden, hidden]
 * @param b_o Output bias [hidden] or NULL
 * @param residual Residual input [hidden]
 * @param hidden Hidden dimension
 * @param num_heads Number of attention heads
 * @param head_dim Head dimension
 */
void fused_output_projection_residual(
    float *output,
    const float *o_all,
    const float *W_o,
    const float *b_o,
    const float *residual,
    int hidden,
    int num_heads,
    int head_dim
);

/*============================================================================
 * Complete Mega-Fused Attention
 *============================================================================*/

/**
 * @brief Complete mega-fused attention block
 *
 * RMSNorm → QKV → RoPE → Flash Attention → OutProj + Residual
 *
 * All intermediates in L1/L2/registers.
 * Single DRAM round-trip: 4KB in + 4KB out.
 *
 * @param output Output tensor [hidden] - single DRAM write
 * @param input Input tensor [hidden] - single DRAM read
 * @param residual Residual input [hidden]
 * @param W_qkv QKV weights [3*hidden, hidden]
 * @param b_qkv QKV bias [3*hidden] or NULL
 * @param W_o Output projection [hidden, hidden]
 * @param b_o Output bias [hidden] or NULL
 * @param kv_cache_k KV cache K [seq, hidden] - updated in place
 * @param kv_cache_v KV cache V [seq, hidden] - updated in place
 * @param rope_cos RoPE cos [max_seq, head_dim/2]
 * @param rope_sin RoPE sin [max_seq, head_dim/2]
 * @param pos Current position
 * @param seq_len Current sequence length
 * @param hidden Hidden dimension
 * @param num_heads Number of heads
 * @param num_kv_heads Number of KV heads
 * @param head_dim Head dimension
 * @param max_seq Maximum sequence length
 * @param eps RMSNorm epsilon
 */
void mega_fused_attention(
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
    int seq_len,
    int hidden,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int max_seq,
    float eps
);

/*============================================================================
 * Statistics and Validation
 *============================================================================*/

/**
 * @brief Report memory savings from mega-fusion
 *
 * @param hidden Hidden dimension
 * @param num_layers Number of layers
 * @param seq_len Sequence length
 */
void fused_kernels_report_stats(
    int hidden,
    int num_layers,
    int seq_len
);

/**
 * @brief Validate cache constraints for fusion
 *
 * @param l1_size L1 cache size
 * @param head_dim Head dimension
 * @param kv_tile_size KV tile size
 * @param bytes_per_elem Element size
 * @return 0 if valid, -1 if working set exceeds cache
 */
int fused_kernels_validate_constraints(
    int l1_size,
    int head_dim,
    int kv_tile_size,
    int bytes_per_elem
);

#endif /* FUSED_KERNELS_H */

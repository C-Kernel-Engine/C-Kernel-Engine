/**
 * @file qk_norm_kernels.c
 * @brief Per-head RMSNorm on Q and K (Qwen3-style QK norm)
 *
 * CK-ENGINE KERNEL RULES:
 * =======================
 * 1. NO malloc/free - memory via bump allocator, pointers passed in
 * 2. NO OpenMP - parallelization at orchestrator/codegen layer
 * 3. API must define: inputs, outputs, workspace, and memory layouts
 * 4. Pure computation - deterministic, no side effects
 *
 * After changes: make test && python unittest/test_qk_norm.py
 *
 * QK Norm normalizes each head's query/key vectors independently before RoPE.
 * This stabilizes Q*K^T dot products before softmax, preventing attention
 * collapse from large magnitude vectors.
 *
 * Why only Q and K, not V?
 *   V does not participate in the attention score computation (Q*K^T).
 *   The softmax saturation problem comes from large Q*K^T values, so only
 *   Q and K magnitudes matter. V is linearly combined after softmax weights
 *   are computed -- normalizing it would change output scale but not fix
 *   attention stability.
 *
 * Data layout after QKV projection (head-major):
 *   Q: [num_heads, num_tokens, head_dim]       contiguous
 *   K: [num_kv_heads, num_tokens, head_dim]     contiguous
 *
 * We treat Q as [num_heads * num_tokens] rows of [head_dim] elements.
 * rmsnorm_forward normalizes each row independently. The gamma weight [head_dim]
 * is shared across all heads (Qwen3 design: one gamma per Q, one per K).
 */

#include <stddef.h>  /* NULL */

/* rmsnorm_forward is declared in ckernel_engine.h */
void rmsnorm_forward(const float *input,
                     const float *gamma,
                     float *output,
                     float *rstd_cache,
                     int tokens,
                     int d_model,
                     int aligned_embed_dim,
                     float eps);

/**
 * Per-head RMSNorm on Q and K.
 *
 * @param q             Q scratch buffer [num_heads * num_tokens * head_dim], in-place
 * @param k             K scratch buffer [num_kv_heads * num_tokens * head_dim], in-place
 * @param q_gamma       Q norm gamma weights [head_dim]
 * @param k_gamma       K norm gamma weights [head_dim]
 * @param num_heads     Number of query heads (e.g. 32 for Qwen3-8B)
 * @param num_kv_heads  Number of KV heads (e.g. 8 for Qwen3-8B with GQA)
 * @param num_tokens    Number of tokens (1 for decode, T for prefill)
 * @param head_dim      Dimension per head (e.g. 128)
 * @param eps           RMSNorm epsilon (e.g. 1e-6)
 *
 * @test unittest/test_qk_norm.py
 */
void qk_norm_forward(float *q, float *k,
                     const float *q_gamma, const float *k_gamma,
                     int num_heads, int num_kv_heads,
                     int num_tokens, int head_dim, float eps)
{
    /* Q norm: [num_heads * num_tokens] rows of [head_dim]
     * Each row is one head's vector for one token. */
    rmsnorm_forward(q, q_gamma, q, NULL,
                    num_heads * num_tokens, head_dim, head_dim, eps);

    /* K norm: [num_kv_heads * num_tokens] rows of [head_dim]
     * Same logic, fewer rows when using GQA. */
    rmsnorm_forward(k, k_gamma, k, NULL,
                    num_kv_heads * num_tokens, head_dim, head_dim, eps);
}

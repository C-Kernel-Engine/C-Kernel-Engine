/**
 * @file v6_simple.c
 * @brief Simplified v6 CLI using only generic kernels
 *
 * This is a minimal v6 implementation that uses:
 * - Generic GEMM (gemm_blocked_serial) instead of quantized kernels
 * - Generic RMSNorm
 * - Precomputed RoPE
 * - OMP parallelization for prefill
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <omp.h>

#include "ckernel_engine.h"

/* Model configuration (hardcoded for Qwen2 0.5B) */
#define MODEL_EMBED_DIM 896
#define MODEL_NUM_LAYERS 24
#define MODEL_NUM_HEADS 14
#define MODEL_NUM_KV_HEADS 2
#define MODEL_HEAD_DIM 64
#define MODEL_INTERMEDIATE_SIZE 4864
#define MODEL_VOCAB_SIZE 128256
#define MODEL_MAX_SEQ_LEN 32768

/* Alignment */
#define ALIGN_EMBED 896
#define ALIGN_HEAD 64
#define MODEL_INTERMEDIATE 4864
#define ALIGN_CONTEXT 32768

/* Simple RMSNorm implementation */
static void simple_rmsnorm(const float *input, const float *gamma, float *output,
                           int tokens, int d_model, float eps) {
    for (int t = 0; t < tokens; t++) {
        const float *in_row = input + t * d_model;
        float *out_row = output + t * d_model;

        /* Compute variance */
        float variance = 0.0f;
        for (int i = 0; i < d_model; i++) {
            variance += in_row[i] * in_row[i];
        }
        variance /= d_model;

        /* Normalize */
        float scale = 1.0f / sqrtf(variance + eps);
        for (int i = 0; i < d_model; i++) {
            out_row[i] = in_row[i] * gamma[i] * scale;
        }
    }
}

/* Simple softmax */
static void softmax(float *x, int n) {
    float max_val = x[0];
    for (int i = 1; i < n; i++) {
        if (x[i] > max_val) max_val = x[i];
    }

    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }

    for (int i = 0; i < n; i++) {
        x[i] /= sum;
    }
}

/* Simple attention (causal) */
static void simple_attention(const float *q, const float *k, const float *v,
                             float *output, int num_heads, int num_kv_heads,
                             int seq_len, int head_dim) {
    int hidden_dim = num_heads * head_dim;

    /* For each head, compute attention */
    for (int h = 0; h < num_heads; h++) {
        const float *q_head = q + h * head_dim;
        const float *k_head = k;  /* KV heads repeated */
        const float *v_head = v;

        /* Repeat K/V for GQA */
        int repeat = num_heads / num_kv_heads;

        float *out_head = output + h * head_dim;

        /* Compute attention scores */
        float scores[MODEL_MAX_SEQ_LEN];
        for (int t = 0; t < seq_len; t++) {
            float score = 0.0f;
            for (int d = 0; d < head_dim; d++) {
                score += q_head[d] * k_head[t * head_dim + d];
            }
            scores[t] = score / sqrtf((float)head_dim);
        }

        /* Causal mask */
        for (int t = 0; t < seq_len; t++) {
            if (t >= seq_len - 1) {
                scores[t] = scores[t];  /* Last token can attend to all */
            } else {
                scores[t] = -1e9f;  /* Mask future tokens */
            }
        }

        /* Softmax */
        softmax(scores, seq_len);

        /* Weighted sum */
        for (int d = 0; d < head_dim; d++) {
            float sum = 0.0f;
            for (int t = 0; t < seq_len; t++) {
                sum += scores[t] * v_head[t * head_dim + d];
            }
            out_head[d] = sum;
        }
    }
}

/* Simple embedding lookup (fp32) */
static void simple_embedding(const int32_t *tokens, int num_tokens,
                            const float *weight, float *output,
                            int vocab_size, int embed_dim) {
    for (int t = 0; t < num_tokens; t++) {
        int token_id = tokens[t];
        if (token_id >= 0 && token_id < vocab_size) {
            memcpy(output + t * embed_dim, weight + token_id * embed_dim,
                   embed_dim * sizeof(float));
        } else {
            memset(output + t * embed_dim, 0, embed_dim * sizeof(float));
        }
    }
}

/* Simple GEMM: output = input @ weight.T (transposed) */
static void gemm_nt(const float *input, const float *weight, float *output,
                    int rows, int cols, int common) {
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            float sum = 0.0f;
            for (int k = 0; k < common; k++) {
                sum += input[r * common + k] * weight[c * common + k];
            }
            output[r * cols + c] = sum;
        }
    }
}

/* Simple SiLU activation */
static void silu(float *x, int n) {
    for (int i = 0; i < n; i++) {
        x[i] = x[i] / (1.0f + expf(-x[i]));
    }
}

/* Simple residual add */
static void residual_add(float *residual, float *addend, int n) {
    for (int i = 0; i < n; i++) {
        residual[i] += addend[i];
    }
}

/* RoPE application (simplified) */
static void apply_rope(float *x, int seq_len, int head_dim) {
    /* Simplified - just identity for now */
    (void)x;
    (void)seq_len;
    (void)head_dim;
}

/* v6 Prefill with OMP parallelization */
void v6_prefill(const float *embed_weight, const int32_t *tokens, int num_tokens,
                float *logits) {
    if (!embed_weight || !tokens || num_tokens <= 0) return;

    /* Allocate buffers */
    const int embed_dim = ALIGN_EMBED;
    const int intermediate = MODEL_INTERMEDIATE;
    const int num_layers = MODEL_NUM_LAYERS;
    const int num_heads = MODEL_NUM_HEADS;
    const int num_kv_heads = MODEL_NUM_KV_HEADS;
    const int head_dim = MODEL_HEAD_DIM;

    /* Per-token hidden states: (num_tokens) x (num_layers + 1) x embed_dim */
    float *hidden = malloc(num_tokens * (num_layers + 1) * embed_dim * sizeof(float));
    if (!hidden) {
        fprintf(stderr, "Failed to allocate hidden states\n");
        return;
    }

    /* Temporary buffers per layer */
    float *q = malloc(num_heads * head_dim * sizeof(float));
    float *k = malloc(num_kv_heads * head_dim * sizeof(float));
    float *v = malloc(num_kv_heads * head_dim * sizeof(float));
    float *attn = malloc(num_heads * head_dim * sizeof(float));
    float *mlp = malloc(intermediate * sizeof(float));

    if (!q || !k || !v || !attn || !mlp) {
        fprintf(stderr, "Failed to allocate temp buffers\n");
        free(hidden);
        free(q);
        free(k);
        free(v);
        free(attn);
        free(mlp);
        return;
    }

    /* Dummy layer weights (in real impl, these come from mapped memory) */
    const float *ln1_gamma = NULL;  /* Would come from weights */
    const float *ln2_gamma = NULL;
    const float *wq = NULL, *wk = NULL, *wv = NULL, *wo = NULL;
    const float *w1 = NULL, *w2 = NULL;

    /* OMP parallel for over tokens */
    #pragma omp parallel for schedule(dynamic, 1)
    for (int t = 0; t < num_tokens; t++) {
        float *h = hidden + t * (num_layers + 1) * embed_dim;

        /* Embedding lookup */
        simple_embedding(tokens + t, 1, embed_weight, h, MODEL_VOCAB_SIZE, embed_dim);

        /* Process through layers */
        for (int layer = 0; layer < num_layers; layer++) {
            float *layer_in = h;
            float *layer_out = h + embed_dim;

            /* RMSNorm */
            simple_rmsnorm(layer_in, ln1_gamma, layer_in, 1, embed_dim, 1e-6f);

            /* QKV projection */
            gemm_nt(layer_in, wq, q, 1, num_heads * head_dim, embed_dim);
            gemm_nt(layer_in, wk, k, 1, num_kv_heads * head_dim, embed_dim);
            gemm_nt(layer_in, wv, v, 1, num_kv_heads * head_dim, embed_dim);

            /* RoPE */
            apply_rope(q, 1, head_dim);
            apply_rope(k, 1, head_dim);

            /* Attention */
            simple_attention(q, k, v, attn, num_heads, num_kv_heads, 1, head_dim);

            /* Output projection */
            gemm_nt(attn, wo, layer_out, 1, embed_dim, num_heads * head_dim);

            /* Residual */
            residual_add(layer_in, layer_out, embed_dim);

            /* RMSNorm before MLP */
            simple_rmsnorm(layer_in, ln2_gamma, layer_in, 1, embed_dim, 1e-6f);

            /* MLP */
            gemm_nt(layer_in, w1, mlp, 1, 2 * intermediate, embed_dim);
            silu(mlp, 2 * intermediate);
            gemm_nt(mlp, w2, layer_out, 1, embed_dim, intermediate);

            /* Residual */
            residual_add(layer_in, layer_out, embed_dim);
        }

        /* Copy to output area */
        memcpy(hidden + t * (num_layers + 1) * embed_dim +
               num_layers * embed_dim, h, embed_dim * sizeof(float));
    }

    /* Final RMSNorm over all tokens */
    float *final_out = malloc(num_tokens * embed_dim * sizeof(float));
    if (final_out) {
        simple_rmsnorm(hidden + num_layers * embed_dim, ln1_gamma, final_out,
                       num_tokens, embed_dim, 1e-6f);

        /* LM head */
        gemm_nt(final_out, embed_weight, logits, num_tokens, MODEL_VOCAB_SIZE, embed_dim);

        free(final_out);
    }

    free(hidden);
    free(q);
    free(k);
    free(v);
    free(attn);
    free(mlp);
}

int main(int argc, char **argv) {
    printf("=== V6 Simple CLI ===\n");
    printf("Generic kernel implementation\n");
    printf("OMP parallelization for prefill\n\n");

    if (argc < 2) {
        printf("Usage: %s <weights.bin> [options]\n", argv[0]);
        printf("\nOptions:\n");
        printf("  -p, --prompt <text>   Input prompt\n");
        printf("  -t, --tokens <n>      Max tokens (default: 50)\n");
        printf("  -h, --help            Show help\n");
        return 1;
    }

    const char *weights_path = argv[1];
    const char *prompt = "Hello";
    int max_tokens = 50;

    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "-p") == 0 || strcmp(argv[i], "--prompt") == 0) {
            prompt = argv[++i];
        } else if (strcmp(argv[i], "-t") == 0 || strcmp(argv[i], "--tokens") == 0) {
            max_tokens = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            printf("Usage: %s <weights.bin> [options]\n", argv[0]);
            return 0;
        }
    }

    printf("Model: Qwen2 0.5B (generic kernels)\n");
    printf("Prompt: %s\n", prompt);
    printf("Max tokens: %d\n", max_tokens);
    printf("\n[Note: This is a simplified v6 implementation using generic kernels]\n");
    printf("[Real weights loading and inference would require full implementation]\n");

    /* Placeholder for actual inference */
    printf("\nAssistant: (v6 placeholder - full implementation pending)\n");

    return 0;
}

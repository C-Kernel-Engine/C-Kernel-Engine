/**
 * test-kernel-parity.cpp
 *
 * Exposes individual ggml kernel operations for parity testing against
 * C-Kernel-Engine implementations.
 *
 * Build as shared library:
 *   g++ -shared -fPIC -o libggml_kernel_test.so \
 *       tests/test-kernel-parity.cpp \
 *       -I ggml/include -I ggml/src \
 *       -L build -lggml -lm -lpthread
 */

#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdio>

// GGML headers
#define GGML_COMMON_DECL_C
#include "ggml-common.h"
#include "ggml.h"
#include "ggml-cpu.h"

// Forward declarations of internal ggml functions we need
extern "C" {
    // Dequantization functions
    void dequantize_row_q4_K(const block_q4_K * x, float * y, int64_t k);
    void dequantize_row_q6_K(const block_q6_K * x, float * y, int64_t k);
    void dequantize_row_q8_K(const block_q8_K * x, float * y, int64_t k);
    void dequantize_row_q4_0(const block_q4_0 * x, float * y, int64_t k);
    void dequantize_row_q5_0(const block_q5_0 * x, float * y, int64_t k);
    void dequantize_row_q8_0(const block_q8_0 * x, float * y, int64_t k);

    // Quantization functions (ref versions from libggml-base.so)
    void quantize_row_q8_K_ref(const float * x, block_q8_K * y, int64_t k);
    void quantize_row_q8_0_ref(const float * x, block_q8_0 * y, int64_t k);

    // Vec dot functions for K-quants (256-block)
    void ggml_vec_dot_q4_K_q8_K(int n, float * s, size_t bs,
                                const void * vx, size_t bx,
                                const void * vy, size_t by, int nrc);
    void ggml_vec_dot_q6_K_q8_K(int n, float * s, size_t bs,
                                const void * vx, size_t bx,
                                const void * vy, size_t by, int nrc);

    // Vec dot functions for legacy quants (32-block)
    void ggml_vec_dot_q5_0_q8_0(int n, float * s, size_t bs,
                                const void * vx, size_t bx,
                                const void * vy, size_t by, int nrc);
    void ggml_vec_dot_q8_0_q8_0(int n, float * s, size_t bs,
                                const void * vx, size_t bx,
                                const void * vy, size_t by, int nrc);
}

extern "C" {

// ============================================================================
// Dequantization Tests
// ============================================================================

/**
 * Test Q4_K dequantization
 * @param src  Input Q4_K blocks (block_q4_K format)
 * @param dst  Output fp32 values
 * @param n    Number of elements to dequantize (must be multiple of QK_K=256)
 */
void test_dequant_q4_k(const void * src, float * dst, int n) {
    dequantize_row_q4_K((const block_q4_K *)src, dst, n);
}

/**
 * Test Q6_K dequantization
 */
void test_dequant_q6_k(const void * src, float * dst, int n) {
    dequantize_row_q6_K((const block_q6_K *)src, dst, n);
}

/**
 * Test Q4_0 dequantization
 */
void test_dequant_q4_0(const void * src, float * dst, int n) {
    dequantize_row_q4_0((const block_q4_0 *)src, dst, n);
}

// ============================================================================
// Quantization Tests
// ============================================================================

/**
 * Test Q8_K quantization (for activations)
 * @param src  Input fp32 values
 * @param dst  Output Q8_K blocks
 * @param n    Number of elements (must be multiple of QK_K=256)
 */
void test_quantize_q8_k(const float * src, void * dst, int n) {
    quantize_row_q8_K_ref(src, (block_q8_K *)dst, n);
}

// ============================================================================
// GEMV (Matrix-Vector) Tests
// ============================================================================

/**
 * Test Q4_K GEMV - dot product of quantized weights and fp32 input
 *
 * Computes: output = weight_q4k . input_f32
 *
 * @param weight_q4k  Q4_K quantized weights [cols]
 * @param input_f32   FP32 input vector [cols]
 * @param output      Output scalar [1]
 * @param cols        Number of columns (must be multiple of QK_K=256)
 */
void test_gemv_q4_k(const void * weight_q4k,
                    const float * input_f32,
                    float * output,
                    int cols) {
    // Allocate Q8_K buffer for quantized activations
    int n_blocks = cols / QK_K;
    block_q8_K * q8_data = new block_q8_K[n_blocks];

    // Quantize input to Q8_K
    quantize_row_q8_K_ref(input_f32, q8_data, cols);

    // Compute dot product
    *output = 0.0f;
    ggml_vec_dot_q4_K_q8_K(cols, output, sizeof(float),
                           weight_q4k, sizeof(block_q4_K),
                           q8_data, sizeof(block_q8_K), 1);

    delete[] q8_data;
}

/**
 * Test Q6_K GEMV
 */
void test_gemv_q6_k(const void * weight_q6k,
                    const float * input_f32,
                    float * output,
                    int cols) {
    int n_blocks = cols / QK_K;
    block_q8_K * q8_data = new block_q8_K[n_blocks];

    quantize_row_q8_K_ref(input_f32, q8_data, cols);

    *output = 0.0f;
    ggml_vec_dot_q6_K_q8_K(cols, output, sizeof(float),
                           weight_q6k, sizeof(block_q6_K),
                           q8_data, sizeof(block_q8_K), 1);

    delete[] q8_data;
}

/**
 * Test Q5_0 GEMV - dot product of quantized weights and fp32 input
 *
 * @param weight_q5_0  Q5_0 quantized weights [cols]
 * @param input_f32    FP32 input vector [cols]
 * @param output       Output scalar [1]
 * @param cols         Number of columns (must be multiple of QK5_0=32)
 */
void test_gemv_q5_0(const void * weight_q5_0,
                    const float * input_f32,
                    float * output,
                    int cols) {
    // Allocate Q8_0 buffer for quantized activations (32-block format)
    int n_blocks = cols / QK8_0;
    block_q8_0 * q8_data = new block_q8_0[n_blocks];

    // Quantize input to Q8_0
    quantize_row_q8_0_ref(input_f32, q8_data, cols);

    // Compute dot product
    *output = 0.0f;
    ggml_vec_dot_q5_0_q8_0(cols, output, sizeof(float),
                           weight_q5_0, sizeof(block_q5_0),
                           q8_data, sizeof(block_q8_0), 1);

    delete[] q8_data;
}

/**
 * Test Q8_0 GEMV - dot product of quantized weights and fp32 input
 *
 * @param weight_q8_0  Q8_0 quantized weights [cols]
 * @param input_f32    FP32 input vector [cols]
 * @param output       Output scalar [1]
 * @param cols         Number of columns (must be multiple of QK8_0=32)
 */
void test_gemv_q8_0(const void * weight_q8_0,
                    const float * input_f32,
                    float * output,
                    int cols) {
    // Allocate Q8_0 buffer for quantized activations
    int n_blocks = cols / QK8_0;
    block_q8_0 * q8_data = new block_q8_0[n_blocks];

    // Quantize input to Q8_0
    quantize_row_q8_0_ref(input_f32, q8_data, cols);

    // Compute dot product
    *output = 0.0f;
    ggml_vec_dot_q8_0_q8_0(cols, output, sizeof(float),
                           weight_q8_0, sizeof(block_q8_0),
                           q8_data, sizeof(block_q8_0), 1);

    delete[] q8_data;
}

/**
 * Test Q5_0 GEMV with FP32 activations
 * Matches CK's approach: dequantize weights to FP32, then FP32 dot product
 */
void test_gemv_q5_0_fp32(const void * weight_q5_0,
                          const float * input_f32,
                          float * output,
                          int cols) {
    // Dequantize single row to FP32
    float * weight_fp32 = new float[cols];
    dequantize_row_q5_0((const block_q5_0 *)weight_q5_0, weight_fp32, cols);

    // FP32 dot product
    float sum = 0.0f;
    for (int i = 0; i < cols; i++) {
        sum += weight_fp32[i] * input_f32[i];
    }
    *output = sum;

    delete[] weight_fp32;
}

/**
 * Test Q8_0 GEMV with FP32 activations
 * Matches CK's approach: dequantize weights to FP32, then FP32 dot product
 */
void test_gemv_q8_0_fp32(const void * weight_q8_0,
                          const float * input_f32,
                          float * output,
                          int cols) {
    // Dequantize single row to FP32
    float * weight_fp32 = new float[cols];
    dequantize_row_q8_0((const block_q8_0 *)weight_q8_0, weight_fp32, cols);

    // FP32 dot product
    float sum = 0.0f;
    for (int i = 0; i < cols; i++) {
        sum += weight_fp32[i] * input_f32[i];
    }
    *output = sum;

    delete[] weight_fp32;
}

// ============================================================================
// Direct Vec Dot Tests (Pre-quantized Q8_0 inputs)
// These test pure kernel accuracy by using the same pre-quantized input
// for both CK and llama.cpp, bypassing quantization differences.
// ============================================================================

/**
 * Test Q5_0 x Q8_0 vec_dot with pre-quantized Q8_0 input
 *
 * @param weight_q5_0  Q5_0 quantized weights [cols]
 * @param input_q8_0   Q8_0 quantized input [cols] (pre-quantized)
 * @param output       Output scalar [1]
 * @param cols         Number of columns (must be multiple of QK8_0=32)
 */
void test_vec_dot_q5_0_q8_0(const void * weight_q5_0,
                             const void * input_q8_0,
                             float * output,
                             int cols) {
    *output = 0.0f;
    ggml_vec_dot_q5_0_q8_0(cols, output, sizeof(float),
                           weight_q5_0, sizeof(block_q5_0),
                           input_q8_0, sizeof(block_q8_0), 1);
}

/**
 * Test Q8_0 x Q8_0 vec_dot with pre-quantized Q8_0 input
 *
 * @param weight_q8_0  Q8_0 quantized weights [cols]
 * @param input_q8_0   Q8_0 quantized input [cols] (pre-quantized)
 * @param output       Output scalar [1]
 * @param cols         Number of columns (must be multiple of QK8_0=32)
 */
void test_vec_dot_q8_0_q8_0(const void * weight_q8_0,
                             const void * input_q8_0,
                             float * output,
                             int cols) {
    *output = 0.0f;
    ggml_vec_dot_q8_0_q8_0(cols, output, sizeof(float),
                           weight_q8_0, sizeof(block_q8_0),
                           input_q8_0, sizeof(block_q8_0), 1);
}

// ============================================================================
// GEMM (Matrix-Matrix) Tests
// ============================================================================

/**
 * Test Q4_K GEMM - batched matrix multiply with quantized weights
 *
 * Computes: output[t,r] = sum_k(weight[r,k] * input[t,k])
 *
 * @param weight_q4k  Q4_K quantized weights [rows, cols]
 * @param input_f32   FP32 input [n_tokens, cols]
 * @param output      FP32 output [n_tokens, rows]
 * @param rows        Number of output rows
 * @param cols        Number of columns (must be multiple of QK_K=256)
 * @param n_tokens    Batch size (number of tokens)
 */
void test_gemm_q4_k(const void * weight_q4k,
                    const float * input_f32,
                    float * output,
                    int rows, int cols, int n_tokens) {
    int n_blocks_per_row = cols / QK_K;
    int weight_row_bytes = n_blocks_per_row * sizeof(block_q4_K);

    // Quantize all input tokens
    block_q8_K * q8_data = new block_q8_K[n_tokens * n_blocks_per_row];
    for (int t = 0; t < n_tokens; t++) {
        quantize_row_q8_K_ref(input_f32 + t * cols,
                              q8_data + t * n_blocks_per_row, cols);
    }

    // Compute output[t,r] for each token and row
    for (int t = 0; t < n_tokens; t++) {
        for (int r = 0; r < rows; r++) {
            float sum = 0.0f;
            const void * w_row = (const char *)weight_q4k + r * weight_row_bytes;
            const block_q8_K * a_row = q8_data + t * n_blocks_per_row;

            ggml_vec_dot_q4_K_q8_K(cols, &sum, sizeof(float),
                                   w_row, sizeof(block_q4_K),
                                   a_row, sizeof(block_q8_K), 1);

            output[t * rows + r] = sum;
        }
    }

    delete[] q8_data;
}

/**
 * Test Q6_K GEMM - batched matrix multiply with quantized weights
 *
 * Computes: output[t,r] = sum_k(weight[r,k] * input[t,k])
 *
 * @param weight_q6k  Q6_K quantized weights [rows, cols]
 * @param input_f32   FP32 input [n_tokens, cols]
 * @param output      FP32 output [n_tokens, rows]
 * @param rows        Number of output rows
 * @param cols        Number of columns (must be multiple of QK_K=256)
 * @param n_tokens    Batch size (number of tokens)
 */
void test_gemm_q6_k(const void * weight_q6k,
                    const float * input_f32,
                    float * output,
                    int rows, int cols, int n_tokens) {
    int n_blocks_per_row = cols / QK_K;
    int weight_row_bytes = n_blocks_per_row * sizeof(block_q6_K);

    // Quantize all input tokens to Q8_K
    block_q8_K * q8_data = new block_q8_K[n_tokens * n_blocks_per_row];
    for (int t = 0; t < n_tokens; t++) {
        quantize_row_q8_K_ref(input_f32 + t * cols,
                              q8_data + t * n_blocks_per_row, cols);
    }

    // Compute output[t,r] for each token and row
    for (int t = 0; t < n_tokens; t++) {
        for (int r = 0; r < rows; r++) {
            float sum = 0.0f;
            const void * w_row = (const char *)weight_q6k + r * weight_row_bytes;
            const block_q8_K * a_row = q8_data + t * n_blocks_per_row;

            ggml_vec_dot_q6_K_q8_K(cols, &sum, sizeof(float),
                                   w_row, sizeof(block_q6_K),
                                   a_row, sizeof(block_q8_K), 1);

            output[t * rows + r] = sum;
        }
    }

    delete[] q8_data;
}

/**
 * Test Q5_0 GEMM - batched matrix multiply with quantized weights (32-element blocks)
 *
 * Computes: output[t,r] = sum_k(weight[r,k] * input[t,k])
 *
 * @param weight_q5_0  Q5_0 quantized weights [rows, cols]
 * @param input_f32    FP32 input [n_tokens, cols]
 * @param output       FP32 output [n_tokens, rows]
 * @param rows         Number of output rows
 * @param cols         Number of columns (must be multiple of QK5_0=32)
 * @param n_tokens     Batch size (number of tokens)
 */
void test_gemm_q5_0(const void * weight_q5_0,
                    const float * input_f32,
                    float * output,
                    int rows, int cols, int n_tokens) {
    int n_blocks_per_row = cols / QK8_0;  // QK5_0 == QK8_0 == 32
    int weight_row_bytes = n_blocks_per_row * sizeof(block_q5_0);

    // Quantize all input tokens to Q8_0
    block_q8_0 * q8_data = new block_q8_0[n_tokens * n_blocks_per_row];
    for (int t = 0; t < n_tokens; t++) {
        quantize_row_q8_0_ref(input_f32 + t * cols,
                              q8_data + t * n_blocks_per_row, cols);
    }

    // Compute output[t,r] for each token and row
    for (int t = 0; t < n_tokens; t++) {
        for (int r = 0; r < rows; r++) {
            float sum = 0.0f;
            const void * w_row = (const char *)weight_q5_0 + r * weight_row_bytes;
            const block_q8_0 * a_row = q8_data + t * n_blocks_per_row;

            ggml_vec_dot_q5_0_q8_0(cols, &sum, sizeof(float),
                                   w_row, sizeof(block_q5_0),
                                   a_row, sizeof(block_q8_0), 1);

            output[t * rows + r] = sum;
        }
    }

    delete[] q8_data;
}

/**
 * Test Q8_0 GEMM - batched matrix multiply with quantized weights (32-element blocks)
 *
 * Computes: output[t,r] = sum_k(weight[r,k] * input[t,k])
 *
 * @param weight_q8_0  Q8_0 quantized weights [rows, cols]
 * @param input_f32    FP32 input [n_tokens, cols]
 * @param output       FP32 output [n_tokens, rows]
 * @param rows         Number of output rows
 * @param cols         Number of columns (must be multiple of QK8_0=32)
 * @param n_tokens     Batch size (number of tokens)
 */
void test_gemm_q8_0(const void * weight_q8_0,
                    const float * input_f32,
                    float * output,
                    int rows, int cols, int n_tokens) {
    int n_blocks_per_row = cols / QK8_0;
    int weight_row_bytes = n_blocks_per_row * sizeof(block_q8_0);

    // Quantize all input tokens to Q8_0
    block_q8_0 * q8_data = new block_q8_0[n_tokens * n_blocks_per_row];
    for (int t = 0; t < n_tokens; t++) {
        quantize_row_q8_0_ref(input_f32 + t * cols,
                              q8_data + t * n_blocks_per_row, cols);
    }

    // Compute output[t,r] for each token and row
    for (int t = 0; t < n_tokens; t++) {
        for (int r = 0; r < rows; r++) {
            float sum = 0.0f;
            const void * w_row = (const char *)weight_q8_0 + r * weight_row_bytes;
            const block_q8_0 * a_row = q8_data + t * n_blocks_per_row;

            ggml_vec_dot_q8_0_q8_0(cols, &sum, sizeof(float),
                                   w_row, sizeof(block_q8_0),
                                   a_row, sizeof(block_q8_0), 1);

            output[t * rows + r] = sum;
        }
    }

    delete[] q8_data;
}

// ============================================================================
// Activation Kernels
// ============================================================================

/**
 * Test RMSNorm
 *
 * Computes: output = (input / rms(input)) * weight
 * where rms(x) = sqrt(mean(x^2) + eps)
 *
 * @param input    Input tensor [n_tokens, dim]
 * @param weight   Normalization weights [dim]
 * @param output   Output tensor [n_tokens, dim]
 * @param n_tokens Number of tokens
 * @param dim      Hidden dimension
 * @param eps      Epsilon for numerical stability
 */
void test_rmsnorm(const float * input,
                  const float * weight,
                  float * output,
                  int n_tokens, int dim, float eps) {
    for (int t = 0; t < n_tokens; t++) {
        const float * x = input + t * dim;
        float * y = output + t * dim;

        // Compute sum of squares
        float sum_sq = 0.0f;
        for (int i = 0; i < dim; i++) {
            sum_sq += x[i] * x[i];
        }

        // RMS = sqrt(mean(x^2) + eps)
        float rms = sqrtf(sum_sq / dim + eps);
        float scale = 1.0f / rms;

        // Normalize and scale by weight
        for (int i = 0; i < dim; i++) {
            y[i] = x[i] * scale * weight[i];
        }
    }
}

/**
 * Test RoPE (Rotary Position Embedding)
 *
 * Applies rotary position embeddings to Q and K tensors.
 * This matches the llama.cpp RoPE implementation (interleaved format).
 *
 * @param q          Query tensor [n_tokens, n_heads * head_dim], modified in-place
 * @param k          Key tensor [n_tokens, n_heads_kv * head_dim], modified in-place
 * @param n_tokens   Number of tokens
 * @param n_heads    Number of query heads
 * @param n_heads_kv Number of key/value heads (can be less for GQA)
 * @param head_dim   Dimension per head
 * @param pos_offset Starting position for RoPE
 * @param theta      RoPE base frequency (typically 10000.0)
 */
void test_rope(float * q, float * k,
               int n_tokens, int n_heads, int n_heads_kv, int head_dim,
               int pos_offset, float theta) {
    // Precompute inverse frequencies
    float * inv_freq = new float[head_dim / 2];
    for (int i = 0; i < head_dim / 2; i++) {
        inv_freq[i] = 1.0f / powf(theta, (float)(2 * i) / head_dim);
    }

    // Apply RoPE to Q
    for (int t = 0; t < n_tokens; t++) {
        int pos = pos_offset + t;
        for (int h = 0; h < n_heads; h++) {
            float * qh = q + t * n_heads * head_dim + h * head_dim;

            for (int i = 0; i < head_dim / 2; i++) {
                float freq = pos * inv_freq[i];
                float cos_val = cosf(freq);
                float sin_val = sinf(freq);

                // Interleaved format: (x0, x1) -> (x0*cos - x1*sin, x0*sin + x1*cos)
                float x0 = qh[i * 2];
                float x1 = qh[i * 2 + 1];
                qh[i * 2]     = x0 * cos_val - x1 * sin_val;
                qh[i * 2 + 1] = x0 * sin_val + x1 * cos_val;
            }
        }
    }

    // Apply RoPE to K
    for (int t = 0; t < n_tokens; t++) {
        int pos = pos_offset + t;
        for (int h = 0; h < n_heads_kv; h++) {
            float * kh = k + t * n_heads_kv * head_dim + h * head_dim;

            for (int i = 0; i < head_dim / 2; i++) {
                float freq = pos * inv_freq[i];
                float cos_val = cosf(freq);
                float sin_val = sinf(freq);

                float x0 = kh[i * 2];
                float x1 = kh[i * 2 + 1];
                kh[i * 2]     = x0 * cos_val - x1 * sin_val;
                kh[i * 2 + 1] = x0 * sin_val + x1 * cos_val;
            }
        }
    }

    delete[] inv_freq;
}

/**
 * Test SwiGLU activation
 *
 * Computes: output = SiLU(gate) * up
 * where input contains [gate, up] concatenated along the last dimension.
 * SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
 *
 * @param gate_up         Input tensor [n_tokens, 2 * intermediate_dim]
 * @param output          Output tensor [n_tokens, intermediate_dim]
 * @param n_tokens        Number of tokens
 * @param intermediate_dim Intermediate dimension (output dim)
 */
void test_swiglu(const float * gate_up,
                 float * output,
                 int n_tokens, int intermediate_dim) {
    for (int t = 0; t < n_tokens; t++) {
        const float * gu = gate_up + t * 2 * intermediate_dim;
        float * out = output + t * intermediate_dim;

        for (int i = 0; i < intermediate_dim; i++) {
            float gate = gu[i];
            float up = gu[intermediate_dim + i];

            // SiLU(gate) * up
            float silu = gate / (1.0f + expf(-gate));
            out[i] = silu * up;
        }
    }
}

/**
 * Test softmax
 *
 * Computes: output[i] = exp(input[i]) / sum(exp(input))
 *
 * @param input   Input tensor [n]
 * @param output  Output tensor [n]
 * @param n       Number of elements
 */
void test_softmax(const float * input, float * output, int n) {
    // Find max for numerical stability
    float max_val = input[0];
    for (int i = 1; i < n; i++) {
        if (input[i] > max_val) max_val = input[i];
    }

    // Compute exp and sum
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        output[i] = expf(input[i] - max_val);
        sum += output[i];
    }

    // Normalize
    for (int i = 0; i < n; i++) {
        output[i] /= sum;
    }
}

// ============================================================================
// Attention Test (for comparing CK-Engine vs llama.cpp)
// ============================================================================

/**
 * @brief Multi-head causal attention for prefill (matches CK-Engine interface)
 *
 * Layout (head-major, matching CK-Engine):
 *   Q: [num_heads, tokens, head_dim]
 *   K: [num_kv_heads, seq_len, head_dim]
 *   V: [num_kv_heads, seq_len, head_dim]
 *   out: [num_heads, tokens, head_dim]
 *
 * Supports GQA (grouped-query attention) where num_heads > num_kv_heads.
 * Causal masking: token t can only attend to positions 0..t (inclusive).
 *
 * @param q           Query [num_heads, tokens, head_dim]
 * @param k           Key [num_kv_heads, seq_len, head_dim]
 * @param v           Value [num_kv_heads, seq_len, head_dim]
 * @param out         Output [num_heads, tokens, head_dim]
 * @param num_heads   Number of query heads
 * @param num_kv_heads Number of key/value heads (for GQA)
 * @param tokens      Number of query tokens
 * @param seq_len     Key/value sequence length (for prefill: seq_len == tokens)
 * @param head_dim    Dimension per head
 */
void test_attention_causal_multihead(const float * q,
                                     const float * k,
                                     const float * v,
                                     float * out,
                                     int num_heads,
                                     int num_kv_heads,
                                     int tokens,
                                     int seq_len,
                                     int head_dim) {
    const float scale = 1.0f / sqrtf((float)head_dim);

    // Allocate temporary buffers (avoid stack overflow for large seq_len)
    float * scores = new float[seq_len];
    float * exp_scores = new float[seq_len];

    // Head strides
    const size_t q_head_stride = (size_t)tokens * (size_t)head_dim;
    const size_t kv_head_stride = (size_t)seq_len * (size_t)head_dim;

    // Process each query head
    for (int h = 0; h < num_heads; h++) {
        // GQA: map query head to KV head
        int kv_h = (num_kv_heads == num_heads) ? h :
                   (int)((long long)h * (long long)num_kv_heads / (long long)num_heads);

        const float * q_head = q + h * q_head_stride;
        const float * k_head = k + kv_h * kv_head_stride;
        const float * v_head = v + kv_h * kv_head_stride;
        float * out_head = out + h * q_head_stride;

        // Process each query token
        for (int t = 0; t < tokens; t++) {
            const float * q_vec = q_head + t * head_dim;
            float * out_vec = out_head + t * head_dim;

            // Causal attention: only attend to positions 0..t (inclusive)
            // For prefill with seq_len == tokens, causal_len = t + 1
            int causal_len = (seq_len == tokens) ? (t + 1) : seq_len;

            // Compute attention scores: q[t] @ k[0:causal_len]^T
            float max_score = -INFINITY;
            for (int s = 0; s < causal_len; s++) {
                const float * k_vec = k_head + s * head_dim;
                float score = 0.0f;
                for (int d = 0; d < head_dim; d++) {
                    score += q_vec[d] * k_vec[d];
                }
                scores[s] = score * scale;
                if (scores[s] > max_score) max_score = scores[s];
            }

            // Softmax: exp and sum
            float sum = 0.0f;
            for (int s = 0; s < causal_len; s++) {
                exp_scores[s] = expf(scores[s] - max_score);
                sum += exp_scores[s];
            }

            // Normalize
            float inv_sum = 1.0f / sum;
            for (int s = 0; s < causal_len; s++) {
                exp_scores[s] *= inv_sum;
            }

            // Weighted sum of values: out = sum(attn_weights * V)
            for (int d = 0; d < head_dim; d++) {
                float result = 0.0f;
                for (int s = 0; s < causal_len; s++) {
                    result += exp_scores[s] * v_head[s * head_dim + d];
                }
                out_vec[d] = result;
            }
        }
    }

    delete[] scores;
    delete[] exp_scores;
}

/**
 * @brief Simple single-head attention (legacy, kept for compatibility)
 */
void test_attention_flash(const float * q,
                         const float * k,
                         const float * v,
                         float * out,
                         int tokens,
                         int head_dim,
                         int seq_len) {
    // Delegate to multi-head version with num_heads=1
    test_attention_causal_multihead(q, k, v, out, 1, 1, tokens, seq_len, head_dim);
}

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * Initialize test library (required by Python bindings)
 * CRITICAL: Must call ggml_cpu_init() to initialize FP16-to-FP32 lookup tables
 * Without this, all K-quant operations return 0 because the lookup table is empty!
 */
void test_init(void) {
    ggml_cpu_init();
}

/**
 * Get Q4_K block size in bytes
 */
int get_block_q4_k_size(void) {
    return sizeof(block_q4_K);
}

/**
 * Get Q6_K block size in bytes
 */
int get_block_q6_k_size(void) {
    return sizeof(block_q6_K);
}

/**
 * Get Q8_K block size in bytes
 */
int get_block_q8_k_size(void) {
    return sizeof(block_q8_K);
}

/**
 * Get Q5_0 block size in bytes
 */
int get_block_q5_0_size(void) {
    return sizeof(block_q5_0);
}

/**
 * Get Q8_0 block size in bytes
 */
int get_block_q8_0_size(void) {
    return sizeof(block_q8_0);
}

/**
 * Get QK_K (elements per K-quant super-block = 256)
 */
int get_qk_k(void) {
    return QK_K;
}

/**
 * Get QK5_0 (elements per Q5_0 block = 32)
 */
int get_qk5_0(void) {
    return QK5_0;
}

/**
 * Get QK8_0 (elements per Q8_0 block = 32)
 */
int get_qk8_0(void) {
    return QK8_0;
}

// ============================================================================
// Fused OutProj + MLP Reference (for parity testing mega_fused_outproj_mlp)
// ============================================================================

/**
 * @brief Reference implementation of fused OutProj + MLP
 *
 * Steps (matches mega_fused_outproj_mlp_prefill):
 * 1. Flatten head-major attn_out to token-major
 * 2. OutProj: attn_flat @ W_o (Q5_0 x Q8_0) → h1
 * 3. Residual: h1 += residual
 * 4. RMSNorm: h1 → ln2_out
 * 5. MLP:
 *    - gate = ln2_out @ W_gate (Q5_0 x Q8_0)
 *    - up = ln2_out @ W_up (Q5_0 x Q8_0)
 *    - hidden = SiLU(gate) * up
 *    - output = hidden @ W2 (Q4_K/Q6_K x Q8_K)
 * 6. Residual: output += h1
 *
 * @param attn_out     Attention output [num_heads, tokens, head_dim] (FP32, head-major)
 * @param residual     Residual input [tokens, embed_dim] (FP32)
 * @param ln2_gamma    RMSNorm gamma [embed_dim] (FP32)
 * @param wo           OutProj weights [embed_dim, embed_dim] (Q5_0)
 * @param w1           MLP W1 weights [2*intermediate, embed_dim] (Q5_0) - gate+up concatenated
 * @param w2           MLP W2 weights [embed_dim, intermediate] (Q4_K or Q6_K)
 * @param output       Output [tokens, embed_dim] (FP32)
 * @param tokens       Number of tokens
 * @param num_heads    Number of attention heads
 * @param head_dim     Dimension per head
 * @param embed_dim    Embedding dimension (= num_heads * head_dim)
 * @param intermediate MLP intermediate dimension
 * @param eps          RMSNorm epsilon
 * @param w2_is_q6k    If true, W2 is Q6_K; if false, W2 is Q4_K
 */
void test_outproj_mlp_fused_q5_0(
    const float * attn_out,
    const float * residual,
    const float * ln2_gamma,
    const void * wo,
    const void * w1,
    const void * w2,
    float * output,
    int tokens,
    int num_heads,
    int head_dim,
    int embed_dim,
    int intermediate,
    float eps,
    int w2_is_q6k)
{
    // Allocate intermediate buffers
    float * attn_flat = new float[tokens * embed_dim];
    float * h1 = new float[tokens * embed_dim];
    float * ln2_out = new float[tokens * embed_dim];
    float * gate = new float[tokens * intermediate];
    float * up = new float[tokens * intermediate];

    // Step 1: Flatten head-major [num_heads, tokens, head_dim] to token-major [tokens, embed_dim]
    // CK layout: attn_out[h, t, d] → attn_flat[t, h * head_dim + d]
    for (int h = 0; h < num_heads; h++) {
        for (int t = 0; t < tokens; t++) {
            for (int d = 0; d < head_dim; d++) {
                int src_idx = h * tokens * head_dim + t * head_dim + d;
                int dst_idx = t * embed_dim + h * head_dim + d;
                attn_flat[dst_idx] = attn_out[src_idx];
            }
        }
    }

    // Step 2: OutProj - attn_flat @ W_o
    // W_o is [embed_dim rows, embed_dim cols] in Q5_0
    test_gemm_q5_0(wo, attn_flat, h1, embed_dim, embed_dim, tokens);

    // Step 3: Residual add - h1 += residual
    for (int i = 0; i < tokens * embed_dim; i++) {
        h1[i] += residual[i];
    }

    // Step 4: RMSNorm - h1 → ln2_out
    test_rmsnorm(h1, ln2_gamma, ln2_out, tokens, embed_dim, eps);

    // Step 5: MLP
    // W1 is [2*intermediate rows, embed_dim cols] - gate and up weights concatenated
    // Gate weights: rows 0 to intermediate-1
    // Up weights: rows intermediate to 2*intermediate-1
    int w1_row_bytes = ((embed_dim + 31) / 32) * sizeof(block_q5_0);
    const void * w_gate = w1;
    const void * w_up = (const char *)w1 + (size_t)intermediate * w1_row_bytes;

    // gate = ln2_out @ W_gate
    test_gemm_q5_0(w_gate, ln2_out, gate, intermediate, embed_dim, tokens);

    // up = ln2_out @ W_up
    test_gemm_q5_0(w_up, ln2_out, up, intermediate, embed_dim, tokens);

    // hidden = SiLU(gate) * up (reuse gate buffer)
    for (int i = 0; i < tokens * intermediate; i++) {
        float g = gate[i];
        float silu = g / (1.0f + expf(-g));
        gate[i] = silu * up[i];
    }

    // output = hidden @ W2 (Q4_K or Q6_K)
    if (w2_is_q6k) {
        test_gemm_q6_k(w2, gate, output, embed_dim, intermediate, tokens);
    } else {
        test_gemm_q4_k(w2, gate, output, embed_dim, intermediate, tokens);
    }

    // Step 6: Residual add - output += h1
    for (int i = 0; i < tokens * embed_dim; i++) {
        output[i] += h1[i];
    }

    delete[] attn_flat;
    delete[] h1;
    delete[] ln2_out;
    delete[] gate;
    delete[] up;
}

/**
 * @brief Get Q5_0 block size for row calculation
 */
int get_q5_0_row_bytes(int cols) {
    return ((cols + 31) / 32) * sizeof(block_q5_0);
}

/**
 * @brief Get Q4_K block size for row calculation
 */
int get_q4_k_row_bytes(int cols) {
    return ((cols + 255) / 256) * sizeof(block_q4_K);
}

/**
 * @brief Get Q6_K block size for row calculation
 */
int get_q6_k_row_bytes(int cols) {
    return ((cols + 255) / 256) * sizeof(block_q6_K);
}

} // extern "C"

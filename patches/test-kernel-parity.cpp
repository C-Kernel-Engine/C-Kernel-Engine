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

} // extern "C"

// Authoritative Q4_K x Q8_K parity against the production llama.cpp CPU graph.
//
// This deliberately tests the v8 public dispatcher, not only a leaf dot
// product. The dispatcher may repack output rows and reuse activation rows;
// those transformations and their accumulation order are part of the
// production numerical contract.

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-cpu/repack.h"
#include "ggml-quants.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

extern "C" {
void quantize_row_q8_k(const float * x, void * y, int k);
void quantize_batch_q8_k_4row_nearest_even(
        const float * x, void * y, int num_rows, int k);
void gemv_q4_k_q8_k(float * y, const void * w, const void * x_q8, int n, int k);
void gemm_nt_q4_k_q8_k_parallel_dispatch(
        const void * a_q8, const void * w, const float * bias, float * out,
        int m, int n, int k);
void gemm_nt_q4_k_q8_k_pairwise_split_min_parallel_dispatch(
        const void * a_q8, const void * w, const float * bias, float * out,
        int m, int n, int k);
void gemv_q4_k_q8_k_repacked_parallel_dispatch(
        float * y, const void * w, const void * x_q8, int n, int k);
void gemv_q4_k_q8_k_parallel_dispatch(
        float * y, const void * w, const void * x_q8, int n, int k);
size_t q4_k_packed_meta_x8_block_size(void);
void pack_q4_k_to_packed_meta_x8(const void * src, void * dst, int n, int k);
void gemm_nt_q4_k_packed_meta_x8_q8_k_superblock_order(
        const void * a_q8, const void * w_packed_x8, const float * bias, float * out,
        int m, int n, int k);
void gemm_nt_q4_k_packed_meta_x16_q8_k_llama_order(
        const void * a_q8, const void * w_packed_x8, const float * bias, float * out,
        int m, int n, int k);
void ck_threadpool_global_destroy(void);
void ck_q4k_packed_weight_cache_clear(void);
void swiglu_forward_ggml(
        const float * input, float * output, int tokens, int dim);
void ggml_vec_swiglu_f32(
        int n, float * output, const float * gate, const float * up);
void ggml_vec_dot_q4_K_q8_K(
        int n, float * s, size_t bs,
        const void * vx, size_t bx,
        const void * vy, size_t by, int nrc);
}

namespace {

static const char * env_or(const char * name, const char * fallback) {
    const char * value = std::getenv(name);
    return value ? value : fallback;
}

static bool env_enabled(const char * name) {
    const char * value = std::getenv(name);
    return value && value[0] != '\0' && std::strcmp(value, "0") != 0;
}

struct case_spec {
    const char * name;
    int m;
    int n;
    int k;
    bool production_dispatch;
    bool decode_dispatch;
    bool with_bias;
};

static float fixture_value(int row, int col, float scale, float phase) {
    const float r = static_cast<float>(row);
    const float c = static_cast<float>(col);
    float value = std::sin(c * 0.017f + r * 0.071f + phase) * scale;
    value += std::cos(c * 0.0031f - r * 0.013f + phase * 0.5f) * (scale * 0.37f);
    return value;
}

static void fill_activations(std::vector<float> & x, int m, int k) {
    for (int row = 0; row < m; ++row) {
        for (int col = 0; col < k; ++col) {
            x[static_cast<size_t>(row) * k + col] = fixture_value(row, col, 0.31f, 0.19f);
        }
        // Exercise signed-max selection and values close to quantization bins.
        for (int block = 0; block < k / QK_K; ++block) {
            const int col = block * QK_K + ((row * 17 + block * 29) % QK_K);
            x[static_cast<size_t>(row) * k + col] =
                    ((row + block) & 1 ? -1.0f : 1.0f) * (0.91f + 0.001f * block);
        }
    }
}

static void fill_weights(std::vector<float> & w, int n, int k) {
    for (int row = 0; row < n; ++row) {
        for (int col = 0; col < k; ++col) {
            w[static_cast<size_t>(row) * k + col] = fixture_value(row, col, 0.13f, 0.47f);
        }
    }
}

static bool compare_bytes(
        const char * label, const uint8_t * ck, const uint8_t * llama, size_t size,
        bool required = true) {
    size_t different = 0;
    size_t first = size;
    for (size_t i = 0; i < size; ++i) {
        if (ck[i] != llama[i]) {
            if (first == size) {
                first = i;
            }
            ++different;
        }
    }
    if (different == 0) {
        std::printf("  %-32s byte_exact (%zu bytes) [PASS]\n", label, size);
        return true;
    }
    std::printf("  %-32s different=%zu/%zu first_byte=%zu ck=%u llama=%u [%s]\n",
            label, different, size, first,
            static_cast<unsigned>(ck[first]), static_cast<unsigned>(llama[first]),
            required ? "FAIL" : "OBSERVED");
    return !required;
}

static bool compare_f32(
        const char * label, const float * ck, const float * llama, int m, int n,
        bool required = true) {
    size_t different = 0;
    size_t first = static_cast<size_t>(m) * n;
    size_t worst = 0;
    float max_abs = 0.0f;
    for (size_t i = 0; i < static_cast<size_t>(m) * n; ++i) {
        if (std::memcmp(&ck[i], &llama[i], sizeof(float)) != 0) {
            if (first == static_cast<size_t>(m) * n) {
                first = i;
            }
            ++different;
        }
        const float diff = std::fabs(ck[i] - llama[i]);
        if (diff > max_abs) {
            max_abs = diff;
            worst = i;
        }
    }
    if (different == 0) {
        std::printf("  %-32s bit_exact (%zu values) [PASS]\n",
                label, static_cast<size_t>(m) * n);
        return true;
    }
    std::printf(
            "  %-32s different=%zu/%zu first=(%zu,%zu) worst=(%zu,%zu) "
            "max_abs=%.9g ck=%.9g llama=%.9g [%s]\n",
            label, different, static_cast<size_t>(m) * n,
            first / n, first % n, worst / n, worst % n,
            max_abs, ck[worst], llama[worst], required ? "FAIL" : "OBSERVED");
    return !required;
}

static bool llama_mul_mat(
        const std::vector<uint8_t> & weights,
        const std::vector<float> & activations,
        std::vector<float> & output,
        int m, int n, int k) {
    const size_t arena_size = 64u * 1024u * 1024u
            + weights.size() + activations.size() * sizeof(float)
            + output.size() * sizeof(float);
    ggml_init_params params = {arena_size, nullptr, false};
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        std::fprintf(stderr, "ggml_init failed for %zu-byte arena\n", arena_size);
        return false;
    }

    ggml_tensor * w = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_K, k, n);
    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, m);
    bool ok = w && x && ggml_nbytes(w) == weights.size();
    if (ok) {
        std::memcpy(ggml_get_data(w), weights.data(), weights.size());
        std::memcpy(ggml_get_data(x), activations.data(), activations.size() * sizeof(float));
        ggml_tensor * y = ggml_mul_mat(ctx, w, x);
        ggml_cgraph * graph = y ? ggml_new_graph(ctx) : nullptr;
        if (!graph) {
            ok = false;
        } else {
            ggml_build_forward_expand(graph, y);
            const int threads = std::max(1, std::atoi(std::getenv("CK_NUM_THREADS")
                    ? std::getenv("CK_NUM_THREADS") : "4"));
            ok = ggml_graph_compute_with_ctx(ctx, graph, threads) == GGML_STATUS_SUCCESS;
            if (ok) {
                std::memcpy(output.data(), ggml_get_data_f32(y), output.size() * sizeof(float));
            }
        }
    }
    ggml_free(ctx);
    return ok;
}

static bool llama_mul_mat_repacked(
        const std::vector<uint8_t> & weights,
        const std::vector<float> & activations,
        std::vector<float> & output,
        int m, int n, int k) {
    ggml_init_params weight_params = {ggml_tensor_overhead() * 2, nullptr, true};
    ggml_context * weight_ctx = ggml_init(weight_params);
    if (!weight_ctx) {
        return false;
    }
    ggml_tensor * w = ggml_new_tensor_2d(weight_ctx, GGML_TYPE_Q4_K, k, n);
    ggml_backend_buffer_type_t buft = ggml_backend_cpu_repack_buffer_type();
    const size_t buffer_size = ggml_backend_buft_get_alloc_size(buft, w);
    ggml_backend_buffer_t buffer = ggml_backend_buft_alloc_buffer(buft, buffer_size);
    bool ok = w && buffer &&
            ggml_backend_tensor_alloc(buffer, w, ggml_backend_buffer_get_base(buffer)) == GGML_STATUS_SUCCESS;
    if (ok) {
        ggml_backend_tensor_set(w, weights.data(), 0, weights.size());
    }

    const size_t arena_size = 64u * 1024u * 1024u
            + activations.size() * sizeof(float) + output.size() * sizeof(float);
    ggml_init_params graph_params = {arena_size, nullptr, false};
    ggml_context * graph_ctx = ok ? ggml_init(graph_params) : nullptr;
    if (graph_ctx) {
        ggml_tensor * x = ggml_new_tensor_2d(graph_ctx, GGML_TYPE_F32, k, m);
        std::memcpy(ggml_get_data(x), activations.data(), activations.size() * sizeof(float));
        ggml_tensor * y = ggml_mul_mat(graph_ctx, w, x);
        ggml_cgraph * graph = y ? ggml_new_graph(graph_ctx) : nullptr;
        if (!graph) {
            ok = false;
        } else {
            ggml_build_forward_expand(graph, y);
            const int threads = std::max(1, std::atoi(std::getenv("CK_NUM_THREADS")
                    ? std::getenv("CK_NUM_THREADS") : "4"));
            ok = ggml_graph_compute_with_ctx(graph_ctx, graph, threads) == GGML_STATUS_SUCCESS;
            if (ok) {
                std::memcpy(output.data(), ggml_get_data_f32(y), output.size() * sizeof(float));
            }
        }
    } else {
        ok = false;
    }

    if (graph_ctx) {
        ggml_free(graph_ctx);
    }
    if (buffer) {
        ggml_backend_buffer_free(buffer);
    }
    ggml_free(weight_ctx);
    return ok;
}

static bool read_f32_file(const char * path, std::vector<float> & values, size_t count) {
    FILE * file = std::fopen(path, "rb");
    if (!file) {
        std::perror(path);
        return false;
    }
    values.resize(count);
    const size_t read = std::fread(values.data(), sizeof(float), count, file);
    std::fclose(file);
    if (read != count) {
        std::fprintf(stderr, "%s: read %zu/%zu float values\n", path, read, count);
        return false;
    }
    return true;
}

static bool read_file_slice(
        const char * path, size_t offset, std::vector<uint8_t> & bytes, size_t size) {
    FILE * file = std::fopen(path, "rb");
    if (!file) {
        std::perror(path);
        return false;
    }
    if (std::fseek(file, static_cast<long>(offset), SEEK_SET) != 0) {
        std::perror(path);
        std::fclose(file);
        return false;
    }
    bytes.resize(size);
    const size_t read = std::fread(bytes.data(), 1, size, file);
    std::fclose(file);
    if (read != size) {
        std::fprintf(stderr, "%s: read %zu/%zu bytes at offset %zu\n", path, read, size, offset);
        return false;
    }
    return true;
}

static bool run_real_artifact_case(void) {
    const char * activation_path = std::getenv("CK_Q4K_Q8K_REAL_ACTIVATIONS_F32");
    const char * weights_path = std::getenv("CK_Q4K_Q8K_REAL_WEIGHTS");
    const char * offset_text = std::getenv("CK_Q4K_Q8K_REAL_WEIGHT_OFFSET");
    const char * bias_offset_text = std::getenv("CK_Q4K_Q8K_REAL_BIAS_OFFSET");
    if (!activation_path && !weights_path && !offset_text) {
        return true;
    }
    if (!activation_path || !weights_path || !offset_text) {
        std::fprintf(stderr,
                "real-artifact mode requires CK_Q4K_Q8K_REAL_ACTIVATIONS_F32, "
                "CK_Q4K_Q8K_REAL_WEIGHTS, and CK_Q4K_Q8K_REAL_WEIGHT_OFFSET\n");
        return false;
    }

    const int m = std::atoi(env_or("CK_Q4K_Q8K_REAL_M", "0"));
    const int n = std::atoi(env_or("CK_Q4K_Q8K_REAL_N", "0"));
    const int k = std::atoi(env_or("CK_Q4K_Q8K_REAL_K", "0"));
    if (m <= 0 || n <= 0 || k <= 0 || (k % QK_K) != 0) {
        std::fprintf(stderr, "invalid real-artifact shape M=%d N=%d K=%d\n", m, n, k);
        return false;
    }

    const size_t q8_row_bytes = static_cast<size_t>(k / QK_K) * sizeof(block_q8_K);
    const size_t q4_row_bytes = static_cast<size_t>(k / QK_K) * sizeof(block_q4_K);
    const size_t weight_offset = static_cast<size_t>(std::strtoull(offset_text, nullptr, 0));
    std::vector<float> activations;
    std::vector<uint8_t> weights;
    std::vector<float> bias;
    if (!read_f32_file(activation_path, activations, static_cast<size_t>(m) * k) ||
        !read_file_slice(weights_path, weight_offset, weights, static_cast<size_t>(n) * q4_row_bytes)) {
        return false;
    }
    if (bias_offset_text) {
        std::vector<uint8_t> bias_bytes;
        const size_t bias_offset = static_cast<size_t>(std::strtoull(bias_offset_text, nullptr, 0));
        if (!read_file_slice(weights_path, bias_offset, bias_bytes, static_cast<size_t>(n) * sizeof(float))) {
            return false;
        }
        bias.resize(static_cast<size_t>(n));
        std::memcpy(bias.data(), bias_bytes.data(), bias_bytes.size());
    }

    std::printf("\nreal_qwen_projection: M=%d N=%d K=%d offset=%zu bias=%s\n",
            m, n, k, weight_offset, bias.empty() ? "no" : "yes");
    std::vector<uint8_t> ck_q8(static_cast<size_t>(m) * q8_row_bytes);
    std::vector<uint8_t> ck_repack_q8(ck_q8.size());
    std::vector<uint8_t> llama_q8(ck_q8.size());
    for (int row = 0; row < m; ++row) {
        const float * src = activations.data() + static_cast<size_t>(row) * k;
        quantize_row_q8_k(src, ck_q8.data() + static_cast<size_t>(row) * q8_row_bytes, k);
        quantize_row_q8_K_ref(src,
                reinterpret_cast<block_q8_K *>(llama_q8.data() + static_cast<size_t>(row) * q8_row_bytes), k);
    }
    quantize_batch_q8_k_4row_nearest_even(
            activations.data(), ck_repack_q8.data(), m, k);
    bool passed = compare_bytes(
            "real Q8_K activation", ck_q8.data(), llama_q8.data(), ck_q8.size());

    std::vector<float> ck_output(static_cast<size_t>(m) * n);
    std::vector<float> llama_output(ck_output.size());
    std::vector<float> llama_leaf(ck_output.size());
    std::vector<float> llama_repacked(ck_output.size());
    std::vector<float> ck_superblock(ck_output.size());
    std::vector<float> ck_x16_llama_order(ck_output.size());
    std::vector<float> ck_x16_repack_q8(ck_output.size());
    std::vector<float> ck_repacked_prefill(ck_output.size());
    std::vector<float> ck_repacked_decode(ck_output.size());
    std::vector<float> production_expected;
    const char * production_expected_path = std::getenv("CK_Q4K_Q8K_REAL_EXPECTED_F32");
    if (production_expected_path &&
        !read_f32_file(production_expected_path, production_expected, ck_output.size())) {
        return false;
    }
    gemm_nt_q4_k_q8_k_parallel_dispatch(
            ck_q8.data(), weights.data(), bias.empty() ? nullptr : bias.data(),
            ck_output.data(), m, n, k);
    for (int row = 0; row < m; ++row) {
        const void * x_row = llama_q8.data() + static_cast<size_t>(row) * q8_row_bytes;
        for (int col = 0; col < n; ++col) {
            const void * w_row = weights.data() + static_cast<size_t>(col) * q4_row_bytes;
            ggml_vec_dot_q4_K_q8_K(
                    k, &llama_leaf[static_cast<size_t>(row) * n + col],
                    0, w_row, 0, x_row, 0, 1);
        }
    }
    if (!llama_mul_mat(weights, activations, llama_output, m, n, k)) {
        return false;
    }
    if (!llama_mul_mat_repacked(weights, activations, llama_repacked, m, n, k)) {
        return false;
    }
    const size_t packed_blocks = static_cast<size_t>((n + 7) / 8) * static_cast<size_t>(k / QK_K);
    std::vector<uint8_t> packed_x8(packed_blocks * q4_k_packed_meta_x8_block_size());
    pack_q4_k_to_packed_meta_x8(weights.data(), packed_x8.data(), n, k);
    gemm_nt_q4_k_packed_meta_x8_q8_k_superblock_order(
            ck_q8.data(), packed_x8.data(), bias.empty() ? nullptr : bias.data(),
            ck_superblock.data(), m, n, k);
    gemm_nt_q4_k_packed_meta_x16_q8_k_llama_order(
            ck_q8.data(), packed_x8.data(), bias.empty() ? nullptr : bias.data(),
            ck_x16_llama_order.data(), m, n, k);
    if ((m % 4) == 0) {
        const int blocks_per_row = k / QK_K;
        std::vector<uint8_t> repack_q8_canonical(
                static_cast<size_t>(m) * q8_row_bytes);
        for (int row0 = 0; row0 < m; row0 += 4) {
            std::vector<block_q8_Kx4> packed_q8(static_cast<size_t>(blocks_per_row));
            ggml_quantize_mat_q8_K_4x8(
                    activations.data() + static_cast<size_t>(row0) * k,
                    packed_q8.data(),
                    k);
            for (int b = 0; b < blocks_per_row; ++b) {
                for (int row_lane = 0; row_lane < 4; ++row_lane) {
                    block_q8_K *dst = reinterpret_cast<block_q8_K *>(
                            repack_q8_canonical.data() +
                            (static_cast<size_t>(row0 + row_lane) * blocks_per_row +
                             static_cast<size_t>(b)) *
                                sizeof(block_q8_K));
                    const block_q8_Kx4 &src = packed_q8[static_cast<size_t>(b)];
                    dst->d = src.d[row_lane];
                    for (int q = 0; q < QK_K; ++q) {
                        const int packed_index =
                                (q / 8) * 32 + row_lane * 8 + (q % 8);
                        dst->qs[q] = src.qs[packed_index];
                    }
                    for (int group = 0; group < QK_K / 16; ++group) {
                        int sum = 0;
                        for (int q = 0; q < 16; ++q) {
                            sum += dst->qs[group * 16 + q];
                        }
                        dst->bsums[group] = static_cast<int16_t>(sum);
                    }
                }
            }
        }
        compare_bytes(
                "real canonical vs repack Q8_K activation",
                ck_q8.data(),
                repack_q8_canonical.data(),
                ck_q8.size());
        compare_bytes(
                "real CK 4-row vs llama repack Q8_K activation",
                ck_repack_q8.data(),
                repack_q8_canonical.data(),
                ck_repack_q8.size());
        for (size_t block_index = 0;
             block_index < ck_repack_q8.size() / sizeof(block_q8_K);
             ++block_index) {
            const block_q8_K *ck_block =
                    reinterpret_cast<const block_q8_K *>(ck_repack_q8.data()) +
                    block_index;
            const block_q8_K *llama_block =
                    reinterpret_cast<const block_q8_K *>(repack_q8_canonical.data()) +
                    block_index;
            if (std::memcmp(ck_block, llama_block, sizeof(*ck_block)) != 0) {
                const int row = static_cast<int>(block_index / blocks_per_row);
                const int block = static_cast<int>(block_index % blocks_per_row);
                std::printf(
                        "    first Q8 block mismatch row=%d block=%d "
                        "ck_d=%.9g llama_d=%.9g d_bits=%08x/%08x\n",
                        row,
                        block,
                        ck_block->d,
                        llama_block->d,
                        *reinterpret_cast<const uint32_t *>(&ck_block->d),
                        *reinterpret_cast<const uint32_t *>(&llama_block->d));
                std::printf(
                        "    implied max_abs ck=%.9g llama=%.9g\n",
                        std::fabs(ck_block->d * 127.0f),
                        std::fabs(llama_block->d * 127.0f));
                for (int q = 0; q < QK_K; ++q) {
                    if (ck_block->qs[q] != llama_block->qs[q]) {
                        std::printf(
                                "    first Q8 quant mismatch q=%d ck=%d llama=%d\n",
                                q,
                                static_cast<int>(ck_block->qs[q]),
                                static_cast<int>(llama_block->qs[q]));
                        break;
                    }
                }
                break;
            }
        }
        gemm_nt_q4_k_packed_meta_x16_q8_k_llama_order(
                ck_repack_q8.data(),
                packed_x8.data(),
                bias.empty() ? nullptr : bias.data(),
                ck_x16_repack_q8.data(),
                m,
                n,
                k);
    }
    gemm_nt_q4_k_q8_k_pairwise_split_min_parallel_dispatch(
            ck_q8.data(), weights.data(), bias.empty() ? nullptr : bias.data(),
            ck_repacked_prefill.data(), m, n, k);
    if (m == 1) {
        gemv_q4_k_q8_k_repacked_parallel_dispatch(
                ck_repacked_decode.data(), weights.data(), ck_q8.data(), n, k);
    }
    if (!bias.empty()) {
        for (int row = 0; row < m; ++row) {
            for (int col = 0; col < n; ++col) {
                const size_t index = static_cast<size_t>(row) * n + col;
                llama_leaf[index] += bias[col];
                llama_output[index] += bias[col];
                llama_repacked[index] += bias[col];
                if (m == 1) {
                    ck_repacked_decode[index] += bias[col];
                }
            }
        }
    }
    passed &= compare_f32(
            "real llama leaf vs graph", llama_leaf.data(), llama_output.data(), m, n);
    passed &= compare_f32(
            "real production graph", ck_output.data(), llama_output.data(), m, n);
    compare_f32(
            "canonical vs repack control", ck_output.data(), llama_repacked.data(), m, n, false);
    if (m == 1) {
        passed &= compare_f32(
                "real repacked decode dispatch",
                ck_repacked_decode.data(), llama_repacked.data(), m, n);
    }
    compare_f32(
            "real superblock-order provider", ck_superblock.data(), llama_repacked.data(), m, n, false);
    passed &= compare_f32(
            "real x16 llama-order provider",
            ck_x16_llama_order.data(), llama_repacked.data(), m, n);
    if ((m % 4) == 0) {
        passed &= compare_f32(
                "real x16 with llama repack Q8_K",
                ck_x16_repack_q8.data(), llama_repacked.data(), m, n);
    }
    passed &= compare_f32(
            "real repacked prefill provider", ck_repacked_prefill.data(), llama_repacked.data(), m, n);
    compare_f32(
            "superblock vs leaf control", ck_superblock.data(), llama_leaf.data(), m, n, false);
    if (!production_expected.empty()) {
        std::printf("\nactual model-graph projection oracle:\n");
        compare_f32(
                "canonical dispatch vs production", ck_output.data(), production_expected.data(), m, n);
        compare_f32(
                "llama leaf vs production", llama_leaf.data(), production_expected.data(), m, n);
        compare_f32(
                "llama graph vs production", llama_output.data(), production_expected.data(), m, n);
        compare_f32(
                "llama repack graph vs production", llama_repacked.data(), production_expected.data(), m, n);
        if (m == 1) {
            compare_f32(
                    "CK repacked decode vs production",
                    ck_repacked_decode.data(), production_expected.data(), m, n);
        }
        compare_f32(
                "CK superblock vs production", ck_superblock.data(), production_expected.data(), m, n);
        compare_f32(
                "CK x16 llama-order vs production",
                ck_x16_llama_order.data(), production_expected.data(), m, n);
        passed &= compare_f32(
                "CK repacked prefill vs production",
                ck_repacked_prefill.data(), production_expected.data(), m, n);
    }
    return passed;
}

static bool run_case(const case_spec & spec) {
    std::printf("\n%s: M=%d N=%d K=%d provider=%s bias=%s\n",
            spec.name, spec.m, spec.n, spec.k,
            spec.production_dispatch ? "gemm_nt_q4_k_q8_k_pairwise_split_min_parallel_dispatch" :
                    (spec.decode_dispatch ? "gemv_q4_k_q8_k_repacked_parallel_dispatch" :
                                            "gemv_q4_k_q8_k_diagnostic_leaf"),
            spec.with_bias ? "yes" : "no");

    std::vector<float> activations(static_cast<size_t>(spec.m) * spec.k);
    std::vector<float> weights_f32(static_cast<size_t>(spec.n) * spec.k);
    fill_activations(activations, spec.m, spec.k);
    fill_weights(weights_f32, spec.n, spec.k);

    const size_t q8_row_bytes = static_cast<size_t>(spec.k / QK_K) * sizeof(block_q8_K);
    const size_t q4_row_bytes = static_cast<size_t>(spec.k / QK_K) * sizeof(block_q4_K);
    std::vector<uint8_t> ck_q8(static_cast<size_t>(spec.m) * q8_row_bytes);
    std::vector<uint8_t> ck_repack_q8(ck_q8.size());
    std::vector<uint8_t> llama_q8(ck_q8.size());
    std::vector<uint8_t> weights(static_cast<size_t>(spec.n) * q4_row_bytes);

    for (int row = 0; row < spec.m; ++row) {
        const float * src = activations.data() + static_cast<size_t>(row) * spec.k;
        quantize_row_q8_k(src, ck_q8.data() + static_cast<size_t>(row) * q8_row_bytes, spec.k);
        quantize_row_q8_K_ref(src,
                reinterpret_cast<block_q8_K *>(llama_q8.data() + static_cast<size_t>(row) * q8_row_bytes),
                spec.k);
    }
    quantize_batch_q8_k_4row_nearest_even(
            activations.data(), ck_repack_q8.data(), spec.m, spec.k);
    for (int row = 0; row < spec.n; ++row) {
        quantize_row_q4_K_ref(weights_f32.data() + static_cast<size_t>(row) * spec.k,
                reinterpret_cast<block_q4_K *>(weights.data() + static_cast<size_t>(row) * q4_row_bytes),
                spec.k);
    }

    bool passed = compare_bytes("Q8_K activation quantizer", ck_q8.data(), llama_q8.data(), ck_q8.size());
    std::vector<float> ck_output(static_cast<size_t>(spec.m) * spec.n);
    std::vector<float> llama_output(ck_output.size());
    std::vector<float> llama_leaf_output(ck_output.size());
    std::vector<float> llama_repacked_output(ck_output.size());
    std::vector<float> ck_repacked_output(ck_output.size());
    std::vector<float> ck_exact_repacked_output(ck_output.size());
    std::vector<float> bias(spec.with_bias ? spec.n : 0);
    for (int col = 0; col < spec.n && spec.with_bias; ++col) {
        bias[col] = fixture_value(0, col, 0.017f, 0.83f);
    }
    for (int row = 0; row < spec.m; ++row) {
        const void * a_row = llama_q8.data() + static_cast<size_t>(row) * q8_row_bytes;
        for (int col = 0; col < spec.n; ++col) {
            const void * w_row = weights.data() + static_cast<size_t>(col) * q4_row_bytes;
            ggml_vec_dot_q4_K_q8_K(
                    spec.k, &llama_leaf_output[static_cast<size_t>(row) * spec.n + col],
                    0, w_row, 0, a_row, 0, 1);
        }
    }
    if (spec.production_dispatch) {
        gemm_nt_q4_k_q8_k_parallel_dispatch(
                ck_q8.data(), weights.data(), spec.with_bias ? bias.data() : nullptr, ck_output.data(),
                spec.m, spec.n, spec.k);
    } else if (spec.decode_dispatch) {
        if (spec.m != 1 || spec.with_bias) {
            std::fprintf(stderr, "decode dispatcher case requires M=1 and no bias\n");
            return false;
        }
        gemv_q4_k_q8_k_parallel_dispatch(
                ck_output.data(), weights.data(), ck_q8.data(), spec.n, spec.k);
    } else {
        gemv_q4_k_q8_k(ck_output.data(), weights.data(), ck_q8.data(), spec.n, spec.k);
    }
    if (!llama_mul_mat(weights, activations, llama_output, spec.m, spec.n, spec.k)) {
        std::fprintf(stderr, "llama.cpp graph execution failed for %s\n", spec.name);
        return false;
    }
    if (spec.production_dispatch || spec.decode_dispatch) {
        if (!llama_mul_mat_repacked(
                weights, activations, llama_repacked_output, spec.m, spec.n, spec.k)) {
            std::fprintf(stderr, "llama.cpp repacked graph failed for %s\n", spec.name);
            return false;
        }
        if (spec.decode_dispatch) {
            gemv_q4_k_q8_k_repacked_parallel_dispatch(
                    ck_repacked_output.data(), weights.data(), ck_q8.data(), spec.n, spec.k);
        } else {
            gemm_nt_q4_k_q8_k_pairwise_split_min_parallel_dispatch(
                    ck_repack_q8.data(), weights.data(), spec.with_bias ? bias.data() : nullptr,
                    ck_repacked_output.data(), spec.m, spec.n, spec.k);
            if ((spec.n % 16) == 0) {
                const size_t packed_blocks =
                        static_cast<size_t>((spec.n + 7) / 8) *
                        static_cast<size_t>(spec.k / QK_K);
                std::vector<uint8_t> packed_x8(
                        packed_blocks * q4_k_packed_meta_x8_block_size());
                pack_q4_k_to_packed_meta_x8(
                        weights.data(), packed_x8.data(), spec.n, spec.k);
                gemm_nt_q4_k_packed_meta_x16_q8_k_llama_order(
                        ck_repack_q8.data(),
                        packed_x8.data(),
                        spec.with_bias ? bias.data() : nullptr,
                        ck_exact_repacked_output.data(),
                        spec.m,
                        spec.n,
                        spec.k);
            }
        }
    }
    if (spec.with_bias) {
        for (int row = 0; row < spec.m; ++row) {
            for (int col = 0; col < spec.n; ++col) {
                const size_t index = static_cast<size_t>(row) * spec.n + col;
                llama_output[index] += bias[col];
                llama_leaf_output[index] += bias[col];
                if (spec.production_dispatch || spec.decode_dispatch) {
                    llama_repacked_output[index] += bias[col];
                }
            }
        }
    }
    if (spec.production_dispatch || spec.decode_dispatch) {
        compare_f32("llama leaf vs graph control",
                llama_leaf_output.data(), llama_output.data(), spec.m, spec.n, false);
        compare_f32("canonical provider control",
                ck_output.data(), llama_output.data(), spec.m, spec.n, false);
        passed &= compare_f32("loaded-model repack output",
                ck_repacked_output.data(), llama_repacked_output.data(), spec.m, spec.n);
        if (spec.production_dispatch && (spec.n % 16) == 0) {
            compare_f32("exact grouped-Q8 x16 provider",
                    ck_exact_repacked_output.data(),
                    llama_repacked_output.data(),
                    spec.m,
                    spec.n,
                    false);
            const int tail_rows = spec.m % 4;
            if (tail_rows != 0) {
                const int tail_start = spec.m - tail_rows;
                compare_f32("tail llama leaf vs repack",
                        llama_leaf_output.data() + static_cast<size_t>(tail_start) * spec.n,
                        llama_repacked_output.data() + static_cast<size_t>(tail_start) * spec.n,
                        tail_rows,
                        spec.n,
                        false);
                compare_f32("tail CK canonical vs repack",
                        ck_output.data() + static_cast<size_t>(tail_start) * spec.n,
                        llama_repacked_output.data() + static_cast<size_t>(tail_start) * spec.n,
                        tail_rows,
                        spec.n,
                        false);
            }
        }
        compare_f32("canonical vs repack control",
                ck_output.data(), llama_repacked_output.data(), spec.m, spec.n, false);
    } else {
        compare_f32("decode leaf primitive control",
                ck_output.data(), llama_leaf_output.data(), spec.m, spec.n, false);
        compare_f32("decode canonical graph control",
                ck_output.data(), llama_output.data(), spec.m, spec.n, false);
    }
    return passed;
}

static bool run_swiglu_case(int tokens, int dim) {
    std::vector<float> input(static_cast<size_t>(tokens) * 2 * dim);
    std::vector<float> ck(static_cast<size_t>(tokens) * dim);
    std::vector<float> llama(ck.size());

    for (int row = 0; row < tokens; ++row) {
        float * gate = input.data() + static_cast<size_t>(row) * 2 * dim;
        float * up = gate + dim;
        for (int col = 0; col < dim; ++col) {
            gate[col] = fixture_value(row, col, 0.73f, 0.31f);
            up[col] = fixture_value(row, col, 0.41f, 0.79f);
        }
        if (dim > 4) {
            gate[(row * 17 + 3) % dim] = -8.125f;
            gate[(row * 29 + 4) % dim] = 7.875f;
        }
    }

    swiglu_forward_ggml(input.data(), ck.data(), tokens, dim);
    for (int row = 0; row < tokens; ++row) {
        const float * gate = input.data() + static_cast<size_t>(row) * 2 * dim;
        const float * up = gate + dim;
        ggml_vec_swiglu_f32(
                dim, llama.data() + static_cast<size_t>(row) * dim, gate, up);
    }

    char label[96];
    std::snprintf(label, sizeof(label), "GGML SwiGLU T=%d D=%d", tokens, dim);
    return compare_bytes(label,
            reinterpret_cast<const uint8_t *>(ck.data()),
            reinterpret_cast<const uint8_t *>(llama.data()),
            ck.size() * sizeof(float), (dim % 8) == 0);
}

} // namespace

int main(int argc, char ** argv) {
    ggml_cpu_init();
    if (env_enabled("CK_REQUIRE_LLAMA_AVX512") && !ggml_cpu_has_avx512()) {
        std::fprintf(stderr,
                "Q4_K x Q8_K oracle requires an AVX-512 llama.cpp build, "
                "but ggml reports avx512=0\n");
        return 2;
    }
    std::printf("llama ISA: avx2=%d avx_vnni=%d avx512=%d avx512_vnni=%d\n",
            ggml_cpu_has_avx2(), ggml_cpu_has_avx_vnni(),
            ggml_cpu_has_avx512(), ggml_cpu_has_avx512_vnni());
    if (!std::getenv("CK_NUM_THREADS")) {
        setenv("CK_NUM_THREADS", "4", 1);
    }

    const bool quick = argc > 1 && std::strcmp(argv[1], "--quick") == 0;
    const case_spec quick_cases[] = {
        {"decode_leaf", 1, 64, 256, false, false, false},
        {"decode_dispatch", 1, 4096, 4096, false, true, false},
        {"qwen3vl_text_before", 5, 1024, 4096, true, false, false},
        {"packed_prefill_threshold", 16, 512, 1024, true, false, true},
        {"packed_prefill_multirow", 33, 1024, 1024, true, false, false},
        {"qwen3vl_text_after", 58, 1024, 4096, true, false, false},
        {"qwen3vl_replay_step45", 59, 1024, 4096, true, false, false},
    };
    const case_spec full_cases[] = {
        {"decode_leaf", 1, 64, 256, false, false, false},
        {"decode_dispatch", 1, 4096, 4096, false, true, false},
        {"qwen3vl_text_before", 5, 1024, 4096, true, false, false},
        {"packed_prefill_threshold", 16, 512, 1024, true, false, true},
        {"packed_prefill_multirow", 33, 1024, 1024, true, false, false},
        {"qwen3vl_text_after", 58, 1024, 4096, true, false, false},
        {"qwen3vl_replay_step45", 59, 1024, 4096, true, false, false},
        {"qwen3vl_qkv_width", 16, 4096, 4096, true, false, false},
        {"qwen3vl_down_k_width", 16, 512, 11008, true, false, false},
    };
    const case_spec * cases = quick ? quick_cases : full_cases;
    const size_t case_count = quick
            ? sizeof(quick_cases) / sizeof(quick_cases[0])
            : sizeof(full_cases) / sizeof(full_cases[0]);

    int quant_failed = 0;
    for (size_t i = 0; i < case_count; ++i) {
        if (!run_case(cases[i])) {
            ++quant_failed;
        }
        ck_q4k_packed_weight_cache_clear();
    }
    if (!run_real_artifact_case()) {
        ++quant_failed;
    }
    const int swiglu_dims[] = {7, 8, 15, 16, 12288};
    int swiglu_differences = 0;
    for (int dim : swiglu_dims) {
        if (!run_swiglu_case(dim == 12288 ? 2 : 3, dim)) {
            ++swiglu_differences;
        }
    }
    ck_threadpool_global_destroy();
    std::printf("\nQ4_K x Q8_K llama.cpp production parity: %s "
                "(%zu Q4 cases, %d failed)\n",
            quant_failed == 0 ? "PASS" : "FAIL", case_count, quant_failed);
    std::printf("SwiGLU ISA diagnostic: %s (%zu cases, %d differences)\n",
            swiglu_differences == 0 ? "PASS" : "OBSERVED",
            sizeof(swiglu_dims) / sizeof(swiglu_dims[0]), swiglu_differences);
    return quant_failed == 0 ? 0 : 1;
}

// Q5_K x Q8_K provider contracts against llama.cpp and decoded arithmetic.

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-cpu/repack.h"
#include "ggml-quants.h"

#include <algorithm>
#include <cmath>
#include <cfloat>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

extern "C" {
void quantize_row_q8_k(const float *, void *, int);
void gemv_q5_k(float *, const void *, const float *, int, int);
void gemm_nt_q5_k(const float *, const void *, const float *, float *, int, int, int);
size_t ck_q5_k_prepared_block_size(void);
void ck_q5_k_prepare_weight(const void *, void *, int, int);
void gemm_nt_q5_k_prepared(const float *, const void *, const float *, float *, int, int, int);
void gemm_nt_q5_k_prepared_m4(const float *, const void *, const float *, float *, int, int, int);
void gemm_nt_q5_k_prepared_q8_m4_nrange(
        const void *, const void *, const float *, float *,
        int, int, int, int, int);
const char *ck_q5_k_provider_name(void);
void ggml_vec_dot_q5_K_q8_K(
        int, float *, size_t, const void *, size_t, const void *, size_t, int);
}

namespace {

struct case_spec { const char * name; int m; int n; int k; };

static float fixture(int row, int col, float scale, float phase) {
    const float r = static_cast<float>(row);
    const float c = static_cast<float>(col);
    float value = std::sin(c * 0.017f + r * 0.071f + phase) * scale;
    value += std::cos(c * 0.0031f - r * 0.013f + phase * 0.5f) * scale * 0.37f;
    return value;
}

static bool compare_bytes(const char * label, const void * ck, const void * llama, size_t size) {
    if (std::memcmp(ck, llama, size) == 0) {
        std::printf("  %-34s byte_exact (%zu bytes) [PASS]\n", label, size);
        return true;
    }
    const auto * a = static_cast<const unsigned char *>(ck);
    const auto * b = static_cast<const unsigned char *>(llama);
    size_t first = size;
    size_t different = 0;
    for (size_t i = 0; i < size; ++i) {
        if (a[i] != b[i]) {
            if (first == size) first = i;
            ++different;
        }
    }
    std::printf("  %-34s different=%zu/%zu first=%zu ck=%u llama=%u [FAIL]\n",
            label, different, size, first, a[first], b[first]);
    return false;
}

static bool compare_f32(const char * label, const float * ck, const float * llama, size_t count) {
    size_t first = count;
    size_t worst = 0;
    size_t different = 0;
    float max_abs = 0.0f;
    for (size_t i = 0; i < count; ++i) {
        if (std::memcmp(ck + i, llama + i, sizeof(float)) != 0) {
            if (first == count) first = i;
            ++different;
        }
        const float diff = std::fabs(ck[i] - llama[i]);
        if (diff > max_abs) {
            max_abs = diff;
            worst = i;
        }
    }
    if (different == 0) {
        std::printf("  %-34s bit_exact (%zu values) [PASS]\n", label, count);
        return true;
    }
    std::printf("  %-34s different=%zu/%zu first=%zu worst=%zu max_abs=%.9g "
                "ck=%.9g llama=%.9g [FAIL]\n",
            label, different, count, first, worst, max_abs, ck[worst], llama[worst]);
    return false;
}

// Decode the serialized operands independently of either dot implementation.
// Integer Q5/Q8 products are exact in both providers; only the superblock
// scale/minimum products and their reduction use FP32 arithmetic. The bound
// below allows 32 FP32 rounding units per absolute scale/minimum term: four
// superblocks, eight live partial sums, and the final horizontal reduction.
// It is deliberately based on operand magnitudes, so cancellation cannot
// conceal an error and the threshold does not depend on this fixture's result.
struct arithmetic_reference {
    std::vector<long double> values;
    std::vector<long double> magnitudes;
};

static void unpack_scale_min(const uint8_t *packed, int group, int &scale, int &minimum) {
    if (group < 4) {
        scale = packed[group] & 63;
        minimum = packed[group + 4] & 63;
    } else {
        scale = (packed[group + 4] & 15) | ((packed[group - 4] >> 6) << 4);
        minimum = (packed[group + 4] >> 4) | ((packed[group] >> 6) << 4);
    }
}

static arithmetic_reference reference_q5_q8(
        const std::vector<unsigned char> &weights,
        const std::vector<unsigned char> &activations, int m, int n, int k) {
    const int blocks = k / QK_K;
    const auto * w = reinterpret_cast<const block_q5_K *>(weights.data());
    const auto * x = reinterpret_cast<const block_q8_K *>(activations.data());
    std::vector<long double> w_values(static_cast<size_t>(n) * k);
    std::vector<long double> w_magnitudes(w_values.size());
    std::vector<long double> x_values(static_cast<size_t>(m) * k);
    for (int row = 0; row < n; ++row) {
        for (int block = 0; block < blocks; ++block) {
            const block_q5_K &wb = w[static_cast<size_t>(row) * blocks + block];
            const long double d = ggml_fp16_to_fp32(wb.data.data.d);
            const long double dmin = ggml_fp16_to_fp32(wb.data.data.dmin);
            for (int group = 0; group < 8; ++group) {
                int scale, minimum;
                unpack_scale_min(wb.scales, group, scale, minimum);
                for (int i = 0; i < 32; ++i) {
                    const int packed_index = (group / 2) * 32 + i;
                    const int low = group & 1 ? wb.qs[packed_index] >> 4
                                              : wb.qs[packed_index] & 15;
                    const int quant = low + ((wb.qh[i] & (1u << group)) ? 16 : 0);
                    const long double positive = d * scale * quant;
                    const long double negative = dmin * minimum;
                    const size_t index = static_cast<size_t>(row) * k + block * QK_K + group * 32 + i;
                    w_values[index] = positive - negative;
                    w_magnitudes[index] = std::fabs(positive) + std::fabs(negative);
                }
            }
        }
    }
    for (int row = 0; row < m; ++row) {
        for (int block = 0; block < blocks; ++block) {
            const block_q8_K &xb = x[static_cast<size_t>(row) * blocks + block];
            for (int i = 0; i < QK_K; ++i) {
                x_values[static_cast<size_t>(row) * k + block * QK_K + i] =
                    static_cast<long double>(xb.d) * xb.qs[i];
            }
        }
    }
    arithmetic_reference result;
    result.values.resize(static_cast<size_t>(m) * n);
    result.magnitudes.resize(result.values.size());
    for (int row = 0; row < m; ++row) {
        for (int col = 0; col < n; ++col) {
            long double sum = 0.0L, magnitude = 0.0L;
            const size_t x_base = static_cast<size_t>(row) * k;
            const size_t w_base = static_cast<size_t>(col) * k;
            for (int i = 0; i < k; ++i) {
                sum += w_values[w_base + i] * x_values[x_base + i];
                magnitude += w_magnitudes[w_base + i] * std::fabs(x_values[x_base + i]);
            }
            const size_t index = static_cast<size_t>(row) * n + col;
            result.values[index] = sum;
            result.magnitudes[index] = magnitude;
        }
    }
    return result;
}

static long double reference_limit(const arithmetic_reference &reference, size_t index) {
    constexpr long double fp32_unit = static_cast<long double>(FLT_EPSILON) / 2;
    return 32 * fp32_unit * reference.magnitudes[index] +
           2 * fp32_unit * std::fabs(reference.values[index]);
}

static bool compare_reference(const char *label, const float *actual,
                              const arithmetic_reference &reference) {
    size_t failed = 0, worst = 0;
    long double max_error = 0.0L, max_ratio = 0.0L, squared = 0.0L;
    for (size_t i = 0; i < reference.values.size(); ++i) {
        const long double error = std::fabs(static_cast<long double>(actual[i]) - reference.values[i]);
        const long double limit = reference_limit(reference, i);
        if (!std::isfinite(actual[i]) || error > limit) ++failed;
        if (error > max_error) { max_error = error; worst = i; }
        if (limit > 0 && error / limit > max_ratio) max_ratio = error / limit;
        squared += error * error;
    }
    std::printf("  %-34s failures=%zu/%zu worst=%zu max_abs=%.9Lg rmse=%.9Lg "
                "max_bound_ratio=%.4Lg [%s]\n", label, failed,
                reference.values.size(), worst, max_error,
                std::sqrt(squared / reference.values.size()), max_ratio,
                failed ? "FAIL" : "PASS");
    return failed == 0;
}

static void report_distinct_reduction(const char *label, const float *a, const float *b,
                                      size_t count) {
    size_t different = 0;
    float max_abs = 0.0f;
    for (size_t i = 0; i < count; ++i) {
        if (std::memcmp(a + i, b + i, sizeof(float)) != 0) ++different;
        max_abs = std::max(max_abs, std::fabs(a[i] - b[i]));
    }
    std::printf("  %-34s different=%zu/%zu max_abs=%.9g [DISTINCT REDUCTION]\n",
                label, different, count, max_abs);
}

static bool llama_graph(
        const std::vector<unsigned char> & weights,
        const std::vector<float> & activations,
        std::vector<float> & output,
        int m, int n, int k, bool repack) {
    ggml_context * weight_ctx = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    ggml_tensor * w = nullptr;
    if (repack) {
        ggml_init_params params = {ggml_tensor_overhead() * 2, nullptr, true};
        weight_ctx = ggml_init(params);
        if (!weight_ctx) return false;
        w = ggml_new_tensor_2d(weight_ctx, GGML_TYPE_Q5_K, k, n);
        ggml_backend_buffer_type_t buft = ggml_backend_cpu_repack_buffer_type();
        buffer = ggml_backend_buft_alloc_buffer(
                buft, ggml_backend_buft_get_alloc_size(buft, w));
        if (!buffer || ggml_backend_tensor_alloc(
                buffer, w, ggml_backend_buffer_get_base(buffer)) != GGML_STATUS_SUCCESS) {
            if (buffer) ggml_backend_buffer_free(buffer);
            ggml_free(weight_ctx);
            return false;
        }
        ggml_backend_tensor_set(w, weights.data(), 0, weights.size());
    }

    const size_t arena = 128u * 1024u * 1024u + weights.size()
            + activations.size() * sizeof(float) + output.size() * sizeof(float);
    ggml_init_params params = {arena, nullptr, false};
    ggml_context * ctx = ggml_init(params);
    if (!ctx) return false;
    if (!repack) {
        w = ggml_new_tensor_2d(ctx, GGML_TYPE_Q5_K, k, n);
        std::memcpy(ggml_get_data(w), weights.data(), weights.size());
    }
    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, m);
    std::memcpy(ggml_get_data(x), activations.data(), activations.size() * sizeof(float));
    ggml_tensor * y = ggml_mul_mat(ctx, w, x);
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, y);
    const int threads = std::max(1, std::atoi(
            std::getenv("CK_NUM_THREADS") ? std::getenv("CK_NUM_THREADS") : "1"));
    const bool ok = ggml_graph_compute_with_ctx(ctx, graph, threads) == GGML_STATUS_SUCCESS;
    if (ok) {
        std::memcpy(output.data(), ggml_get_data_f32(y), output.size() * sizeof(float));
    }
    ggml_free(ctx);
    if (buffer) ggml_backend_buffer_free(buffer);
    if (weight_ctx) ggml_free(weight_ctx);
    return ok;
}

static bool run_case(const case_spec & spec) {
    const char *tiled_setting = std::getenv("GGML_CPU_TILED_MM");
    const bool tiled_disabled = tiled_setting && std::strcmp(tiled_setting, "0") == 0;
    std::printf("\n%s M=%d N=%d K=%d threads=%s CK_provider=%s llama_tiled_setting=%s\n",
            spec.name, spec.m, spec.n, spec.k,
            std::getenv("CK_NUM_THREADS") ? std::getenv("CK_NUM_THREADS") : "1",
            ck_q5_k_provider_name(), tiled_setting ? tiled_setting : "default");
    std::vector<float> activations(static_cast<size_t>(spec.m) * spec.k);
    std::vector<float> weights_f32(static_cast<size_t>(spec.n) * spec.k);
    for (int row = 0; row < spec.m; ++row) {
        for (int col = 0; col < spec.k; ++col) {
            float value = fixture(row, col, 0.31f, 0.19f);
            if (std::strcmp(spec.name, "q8_rounding_boundary") == 0) {
                const int code = (col % 251) - 125;
                value = (static_cast<float>(code) + 0.5f) / 127.0f;
                if (col % 256 == 0) value = 1.0f;
            }
            activations[static_cast<size_t>(row) * spec.k + col] = value;
        }
    }
    for (int row = 0; row < spec.n; ++row) {
        for (int col = 0; col < spec.k; ++col) {
            weights_f32[static_cast<size_t>(row) * spec.k + col] =
                    fixture(row, col, 0.13f, 0.47f);
        }
    }

    const size_t q8_row = static_cast<size_t>(spec.k / QK_K) * sizeof(block_q8_K);
    const size_t q5_row = static_cast<size_t>(spec.k / QK_K) * sizeof(block_q5_K);
    std::vector<unsigned char> ck_q8(static_cast<size_t>(spec.m) * q8_row);
    std::vector<unsigned char> llama_q8(ck_q8.size());
    std::vector<unsigned char> weights(static_cast<size_t>(spec.n) * q5_row);
    for (int row = 0; row < spec.m; ++row) {
        const float * input = activations.data() + static_cast<size_t>(row) * spec.k;
        quantize_row_q8_k(input, ck_q8.data() + static_cast<size_t>(row) * q8_row, spec.k);
        quantize_row_q8_K_ref(input, reinterpret_cast<block_q8_K *>(
                llama_q8.data() + static_cast<size_t>(row) * q8_row), spec.k);
    }
    for (int row = 0; row < spec.n; ++row) {
        quantize_row_q5_K_ref(weights_f32.data() + static_cast<size_t>(row) * spec.k,
                reinterpret_cast<block_q5_K *>(weights.data() + static_cast<size_t>(row) * q5_row), spec.k);
    }

    bool pass = compare_bytes(
            "Q8_K activation quantizer", ck_q8.data(), llama_q8.data(), ck_q8.size());
    std::vector<float> leaf(static_cast<size_t>(spec.m) * spec.n);
    std::vector<float> canonical(leaf.size());
    std::vector<float> production(leaf.size());
    std::vector<float> ck(leaf.size());
    std::vector<float> prepared_output(leaf.size());
    std::vector<float> prepared_m4_output(leaf.size());
    std::vector<float> prepared_nsplit_output(leaf.size());
    for (int row = 0; row < spec.m; ++row) {
        for (int col = 0; col < spec.n; ++col) {
            ggml_vec_dot_q5_K_q8_K(spec.k, &leaf[static_cast<size_t>(row) * spec.n + col], 0,
                    weights.data() + static_cast<size_t>(col) * q5_row, 0,
                    llama_q8.data() + static_cast<size_t>(row) * q8_row, 0, 1);
        }
    }
    if (spec.m == 1) {
        gemv_q5_k(ck.data(), weights.data(), activations.data(), spec.n, spec.k);
    } else {
        gemm_nt_q5_k(activations.data(), weights.data(), nullptr,
                ck.data(), spec.m, spec.n, spec.k);
    }
    const size_t prepared_blocks = static_cast<size_t>(spec.n) * spec.k / QK_K;
    std::vector<unsigned char> prepared_weights(
            prepared_blocks * ck_q5_k_prepared_block_size());
    ck_q5_k_prepare_weight(
            weights.data(), prepared_weights.data(), spec.n, spec.k);
    gemm_nt_q5_k_prepared(
            activations.data(), prepared_weights.data(), nullptr,
            prepared_output.data(), spec.m, spec.n, spec.k);
    gemm_nt_q5_k_prepared_m4(
            activations.data(), prepared_weights.data(), nullptr,
            prepared_m4_output.data(), spec.m, spec.n, spec.k);
    const int n_mid = spec.n / 2;
    if (n_mid > 0) {
        gemm_nt_q5_k_prepared_q8_m4_nrange(
            ck_q8.data(), prepared_weights.data(), nullptr,
            prepared_nsplit_output.data(), spec.m, spec.n, spec.k, 0, n_mid);
    }
    gemm_nt_q5_k_prepared_q8_m4_nrange(
        ck_q8.data(), prepared_weights.data(), nullptr,
        prepared_nsplit_output.data(), spec.m, spec.n, spec.k, n_mid, spec.n);
    if (!llama_graph(weights, activations, canonical, spec.m, spec.n, spec.k, false)) return false;
    const bool llama_q5_repack_selected = ggml_cpu_has_neon();
    if (llama_q5_repack_selected) {
        if (!llama_graph(weights, activations, production, spec.m, spec.n, spec.k, true)) return false;
    } else {
        production = canonical;
        std::printf("  %-34s x86 Q5 uses canonical graph [SKIP]\n", "llama Q5 repack graph");
    }
    const arithmetic_reference reference = reference_q5_q8(
        weights, llama_q8, spec.m, spec.n, spec.k);
    pass &= compare_reference("llama leaf vs math", leaf.data(), reference);
    pass &= compare_reference("llama graph vs math", canonical.data(), reference);
    if (llama_q5_repack_selected) {
        pass &= compare_reference("llama repack vs math", production.data(), reference);
    }
    pass &= compare_reference("CK adapter vs math", ck.data(), reference);
    if (spec.m < 8 || tiled_disabled) {
        // Below the pinned llama.cpp tiled-matmul admission floor, the graph
        // calls the same leaf reduction. Different execution strategies are
        // compared to the mathematical reference below instead.
        pass &= compare_f32("llama leaf vs canonical graph", leaf.data(), canonical.data(), leaf.size());
    } else {
        report_distinct_reduction("llama leaf vs canonical graph",
                                  leaf.data(), canonical.data(), leaf.size());
    }
    if (llama_q5_repack_selected) {
        report_distinct_reduction("llama canonical vs repack graph",
                                  canonical.data(), production.data(), leaf.size());
    }
    if (spec.m == 1 && spec.k == QK_K) {
        pass &= compare_f32("CK adapter vs llama production", ck.data(), production.data(), leaf.size());
    } else {
        report_distinct_reduction("CK adapter vs llama production",
                                  ck.data(), production.data(), leaf.size());
    }
    pass &= compare_f32("CK prepared vs established", prepared_output.data(), ck.data(), leaf.size());
    pass &= compare_f32("CK prepared M4 vs established", prepared_m4_output.data(), ck.data(), leaf.size());
    pass &= compare_f32("CK prepared N-split vs established", prepared_nsplit_output.data(), ck.data(), leaf.size());
    const float corrupted = ck[0] + static_cast<float>(
        std::max(0.01L, 4 * reference_limit(reference, 0)));
    const bool detected = std::fabs(static_cast<long double>(corrupted) - reference.values[0]) >
                          reference_limit(reference, 0);
    std::printf("  %-34s wrong output %s\n", "math negative control",
                detected ? "[DETECTED]" : "[FAIL]");
    pass &= detected;
    std::vector<unsigned char> wrong_scale_q8 = llama_q8;
    reinterpret_cast<block_q8_K *>(wrong_scale_q8.data())[0].d *= 1.25f;
    size_t scale_detected_at = static_cast<size_t>(spec.n);
    for (int col = 0; col < spec.n; ++col) {
        float wrong_scale_output;
        ggml_vec_dot_q5_K_q8_K(spec.k, &wrong_scale_output, 0,
                weights.data() + static_cast<size_t>(col) * q5_row, 0,
                wrong_scale_q8.data(), 0, 1);
        if (std::fabs(static_cast<long double>(wrong_scale_output) - reference.values[col]) >
                reference_limit(reference, col)) {
            scale_detected_at = static_cast<size_t>(col);
            break;
        }
    }
    std::printf("  %-34s changed Q8_K scale %s (column=%zu)\n",
                "operand negative control", scale_detected_at < static_cast<size_t>(spec.n)
                ? "[DETECTED]" : "[FAIL]", scale_detected_at);
    pass &= scale_detected_at < static_cast<size_t>(spec.n);
    return pass;
}

} // namespace

int main(int argc, char ** argv) {
    ggml_cpu_init();
    const bool quick = argc > 1 && std::strcmp(argv[1], "--quick") == 0;
    const case_spec quick_cases[] = {
        {"decode_leaf", 1, 64, 256},
        {"q8_rounding_boundary", 1, 64, 1024},
        {"qwen35_recurrent_decode", 1, 6144, 1024},
        {"short_prefill", 7, 512, 1024},
        {"qwen35_recurrent_prefill", 33, 6144, 1024},
    };
    const case_spec full_cases[] = {
        {"decode_leaf", 1, 64, 256},
        {"q8_rounding_boundary", 1, 64, 1024},
        {"qwen35_recurrent_decode", 1, 6144, 1024},
        {"qwen35_recurrent_prefill", 33, 6144, 1024},
    };
    const case_spec * cases = quick ? quick_cases : full_cases;
    const size_t count = quick ? sizeof(quick_cases) / sizeof(quick_cases[0])
                               : sizeof(full_cases) / sizeof(full_cases[0]);
    int failed = 0;
    for (size_t i = 0; i < count; ++i) {
        if (!run_case(cases[i])) ++failed;
    }
    std::printf("\nQ5_K x Q8_K llama.cpp production parity: %s (%zu cases, %d failed)\n",
            failed ? "FAIL" : "PASS", count, failed);
    return failed ? 1 : 0;
}

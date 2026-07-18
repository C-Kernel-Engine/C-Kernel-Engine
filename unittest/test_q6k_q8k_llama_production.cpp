// Authoritative Q6_K x Q8_K parity against llama.cpp's CPU leaf and graph.

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
#include <fstream>
#include <vector>

extern "C" {
void quantize_row_q8_k(const float *, void *, int);
void gemv_q6_k_q8_k(float *, const void *, const void *, int, int);
void gemv_q6_k_q8_k_parallel_dispatch(float *, const void *, const void *, int, int);
void gemm_nt_q6_k_q8_k_parallel_dispatch(
        const void *, const void *, const float *, float *, int, int, int);
const char *ck_q6_k_q8_k_provider_name(void);
void ck_threadpool_global_destroy(void);
void ggml_vec_dot_q6_K_q8_K(
        int, float *, size_t, const void *, size_t, const void *, size_t, int);
}

namespace {

struct case_spec { const char *name; int m; int n; int k; };

static bool env_enabled(const char *name) {
    const char *value = std::getenv(name);
    return value && value[0] != '\0' && std::strcmp(value, "0") != 0;
}

static float fixture(int row, int col, float scale, float phase) {
    const float r = static_cast<float>(row);
    const float c = static_cast<float>(col);
    return std::sin(c * 0.017f + r * 0.071f + phase) * scale
            + std::cos(c * 0.0031f - r * 0.013f + phase * 0.5f) * scale * 0.37f;
}

static bool compare_bytes(const char *label, const void *a, const void *b, size_t size) {
    if (std::memcmp(a, b, size) == 0) {
        std::printf("  %-34s byte_exact (%zu bytes) [PASS]\n", label, size);
        return true;
    }
    const auto *aa = static_cast<const unsigned char *>(a);
    const auto *bb = static_cast<const unsigned char *>(b);
    size_t first = 0, different = 0;
    for (size_t i = 0; i < size; ++i) {
        if (aa[i] != bb[i]) { if (different++ == 0) first = i; }
    }
    std::printf("  %-34s different=%zu/%zu first=%zu ck=%u llama=%u [FAIL]\n",
            label, different, size, first, aa[first], bb[first]);
    return false;
}

static bool compare_f32(const char *label, const float *a, const float *b, size_t count) {
    size_t different = 0, first = count, worst = 0;
    float max_abs = 0.0f;
    for (size_t i = 0; i < count; ++i) {
        if (std::memcmp(a + i, b + i, sizeof(float)) != 0) {
            if (first == count) first = i;
            ++different;
        }
        const float d = std::fabs(a[i] - b[i]);
        if (d > max_abs) { max_abs = d; worst = i; }
    }
    if (different == 0) {
        std::printf("  %-34s bit_exact (%zu values) [PASS]\n", label, count);
        return true;
    }
    std::printf("  %-34s different=%zu/%zu first=%zu worst=%zu max_abs=%.9g "
                "ck=%.9g llama=%.9g [FAIL]\n",
            label, different, count, first, worst, max_abs, a[worst], b[worst]);
    return false;
}

static float f32_from_bits(uint32_t bits) {
    float value;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

static bool run_q8_rounding_boundary_case() {
    std::printf("\nq8_rounding_boundary K=%d\n", QK_K);
    std::vector<float> input(QK_K, 0.0f);
    input[119] = f32_from_bits(0x3e9ade7bU);
    input[204] = f32_from_bits(0xbe508645U);

    block_q8_K ck = {};
    block_q8_K llama = {};
    quantize_row_q8_k(input.data(), &ck, QK_K);
    quantize_row_q8_K_ref(input.data(), &llama, QK_K);
    return compare_bytes("Q8_K non-contracted rounding", &ck, &llama, sizeof(ck));
}

static std::vector<unsigned char> read_bytes(const char *path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) return {};
    const std::streamsize size = input.tellg();
    if (size < 0) return {};
    std::vector<unsigned char> data(static_cast<size_t>(size));
    input.seekg(0);
    input.read(reinterpret_cast<char *>(data.data()), size);
    return input ? data : std::vector<unsigned char>{};
}

static bool llama_graph(const std::vector<unsigned char> &weights,
        const std::vector<float> &activations, std::vector<float> &output,
        int m, int n, int k, bool request_repack) {
    ggml_context *weight_ctx = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    ggml_tensor *w = nullptr;
    if (request_repack) {
        ggml_init_params p = {ggml_tensor_overhead() * 2, nullptr, true};
        weight_ctx = ggml_init(p);
        w = ggml_new_tensor_2d(weight_ctx, GGML_TYPE_Q6_K, k, n);
        ggml_backend_buffer_type_t buft = ggml_backend_cpu_repack_buffer_type();
        buffer = ggml_backend_buft_alloc_buffer(buft, ggml_backend_buft_get_alloc_size(buft, w));
        if (!buffer || ggml_backend_tensor_alloc(
                buffer, w, ggml_backend_buffer_get_base(buffer)) != GGML_STATUS_SUCCESS) return false;
        ggml_backend_tensor_set(w, weights.data(), 0, weights.size());
    }

    const size_t arena = 64u * 1024u * 1024u + weights.size()
            + activations.size() * sizeof(float) + output.size() * sizeof(float);
    ggml_init_params p = {arena, nullptr, false};
    ggml_context *ctx = ggml_init(p);
    if (!request_repack) {
        w = ggml_new_tensor_2d(ctx, GGML_TYPE_Q6_K, k, n);
        std::memcpy(ggml_get_data(w), weights.data(), weights.size());
    }
    ggml_tensor *x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, m);
    std::memcpy(ggml_get_data(x), activations.data(), activations.size() * sizeof(float));
    ggml_tensor *y = ggml_mul_mat(ctx, w, x);
    ggml_cgraph *graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, y);
    const int threads = std::max(1, std::atoi(std::getenv("CK_NUM_THREADS")
            ? std::getenv("CK_NUM_THREADS") : "1"));
    const bool ok = ggml_graph_compute_with_ctx(ctx, graph, threads) == GGML_STATUS_SUCCESS;
    if (ok) std::memcpy(output.data(), ggml_get_data_f32(y), output.size() * sizeof(float));
    ggml_free(ctx);
    if (buffer) ggml_backend_buffer_free(buffer);
    if (weight_ctx) ggml_free(weight_ctx);
    return ok;
}

static bool run_case(const case_spec &spec) {
    std::printf("\n%s M=%d N=%d K=%d ck_provider=%s threads=%s\n",
            spec.name, spec.m, spec.n, spec.k, ck_q6_k_q8_k_provider_name(),
            std::getenv("CK_NUM_THREADS") ? std::getenv("CK_NUM_THREADS") : "1");
    std::vector<float> activations(static_cast<size_t>(spec.m) * spec.k);
    std::vector<float> weights_f32(static_cast<size_t>(spec.n) * spec.k);
    for (int r = 0; r < spec.m; ++r) for (int c = 0; c < spec.k; ++c)
        activations[static_cast<size_t>(r) * spec.k + c] = fixture(r, c, 0.31f, 0.19f);
    for (int r = 0; r < spec.n; ++r) for (int c = 0; c < spec.k; ++c)
        weights_f32[static_cast<size_t>(r) * spec.k + c] = fixture(r, c, 0.13f, 0.47f);

    const size_t q8_row = static_cast<size_t>(spec.k / QK_K) * sizeof(block_q8_K);
    const size_t q6_row = static_cast<size_t>(spec.k / QK_K) * sizeof(block_q6_K);
    std::vector<unsigned char> ck_q8(static_cast<size_t>(spec.m) * q8_row);
    std::vector<unsigned char> llama_q8(ck_q8.size());
    std::vector<unsigned char> weights(static_cast<size_t>(spec.n) * q6_row);
    for (int r = 0; r < spec.m; ++r) {
        const float *src = activations.data() + static_cast<size_t>(r) * spec.k;
        quantize_row_q8_k(src, ck_q8.data() + static_cast<size_t>(r) * q8_row, spec.k);
        quantize_row_q8_K_ref(src, reinterpret_cast<block_q8_K *>(
                llama_q8.data() + static_cast<size_t>(r) * q8_row), spec.k);
    }
    for (int r = 0; r < spec.n; ++r) {
        quantize_row_q6_K_ref(weights_f32.data() + static_cast<size_t>(r) * spec.k,
                reinterpret_cast<block_q6_K *>(weights.data() + static_cast<size_t>(r) * q6_row), spec.k);
    }

    bool pass = compare_bytes("Q8_K activation quantizer", ck_q8.data(), llama_q8.data(), ck_q8.size());
    std::vector<float> ck(static_cast<size_t>(spec.m) * spec.n);
    std::vector<float> leaf(ck.size()), canonical(ck.size()), repack(ck.size());
    for (int r = 0; r < spec.m; ++r) for (int c = 0; c < spec.n; ++c) {
        ggml_vec_dot_q6_K_q8_K(spec.k, &leaf[static_cast<size_t>(r) * spec.n + c], 0,
                weights.data() + static_cast<size_t>(c) * q6_row, 0,
                llama_q8.data() + static_cast<size_t>(r) * q8_row, 0, 1);
    }
    if (spec.m == 1) {
        gemv_q6_k_q8_k_parallel_dispatch(ck.data(), weights.data(), ck_q8.data(), spec.n, spec.k);
    } else {
        gemm_nt_q6_k_q8_k_parallel_dispatch(
                ck_q8.data(), weights.data(), nullptr, ck.data(), spec.m, spec.n, spec.k);
    }
    if (!llama_graph(weights, activations, canonical, spec.m, spec.n, spec.k, false)) return false;
    const bool llama_q6_repack_selected = ggml_cpu_has_neon();
    if (llama_q6_repack_selected) {
        if (!llama_graph(weights, activations, repack, spec.m, spec.n, spec.k, true)) return false;
    } else {
        repack = canonical;
        std::printf("  %-34s x86 Q6 uses canonical graph [SKIP]\n", "llama Q6 repack graph");
    }

    pass &= compare_f32("llama leaf vs canonical graph", leaf.data(), canonical.data(), ck.size());
    if (llama_q6_repack_selected) {
        pass &= compare_f32("llama canonical vs repack graph", canonical.data(), repack.data(), ck.size());
    }
    pass &= compare_f32(spec.m == 1 ? "CK decode vs llama production" :
            "CK prefill vs llama production", ck.data(), repack.data(), ck.size());
    return pass;
}

static bool run_xray_artifact_case() {
    const char *input_path = std::getenv("CK_Q6_XRAY_INPUT_F32");
    const char *weights_path = std::getenv("CK_Q6_XRAY_WEIGHTS_Q6_K");
    const char *output_path = std::getenv("CK_Q6_XRAY_LLAMA_OUTPUT_F32");
    if (!input_path && !weights_path && !output_path) return true;
    if (!input_path || !weights_path || !output_path) {
        std::fprintf(stderr, "all CK_Q6_XRAY_* paths are required together\n");
        return false;
    }

    const int k = std::atoi(std::getenv("CK_Q6_XRAY_K") ? std::getenv("CK_Q6_XRAY_K") : "4096");
    const int n = std::atoi(std::getenv("CK_Q6_XRAY_N") ? std::getenv("CK_Q6_XRAY_N") : "1024");
    const auto input_bytes = read_bytes(input_path);
    const auto weights = read_bytes(weights_path);
    const auto output_bytes = read_bytes(output_path);
    const size_t q6_row = static_cast<size_t>(k / QK_K) * sizeof(block_q6_K);
    if (input_bytes.size() != static_cast<size_t>(k) * sizeof(float)
            || weights.size() != static_cast<size_t>(n) * q6_row
            || output_bytes.size() != static_cast<size_t>(n) * sizeof(float)) {
        std::fprintf(stderr, "invalid Q6 X-ray artifact extents: input=%zu weights=%zu output=%zu\n",
                input_bytes.size(), weights.size(), output_bytes.size());
        return false;
    }

    std::vector<float> input(k), expected(n), leaf(n), graph(n), ck(n);
    std::memcpy(input.data(), input_bytes.data(), input_bytes.size());
    std::memcpy(expected.data(), output_bytes.data(), output_bytes.size());
    std::vector<block_q8_K> q8(static_cast<size_t>(k / QK_K));
    quantize_row_q8_k(input.data(), q8.data(), k);
    for (int row = 0; row < n; ++row) {
        ggml_vec_dot_q6_K_q8_K(k, &leaf[row], 0,
                weights.data() + static_cast<size_t>(row) * q6_row, 0, q8.data(), 0, 1);
    }
    gemv_q6_k_q8_k_parallel_dispatch(ck.data(), weights.data(), q8.data(), n, k);
    if (!llama_graph(weights, input, graph, 1, n, k, false)) return false;

    std::printf("\nxray_artifact M=1 N=%d K=%d\n", n, k);
    bool pass = compare_f32("llama leaf vs graph", leaf.data(), graph.data(), n);
    pass &= compare_f32("CK vs llama graph", ck.data(), graph.data(), n);
    pass &= compare_f32("llama graph vs captured V", graph.data(), expected.data(), n);
    return pass;
}

} // namespace

int main(int argc, char **argv) {
    ggml_cpu_init();
    if (env_enabled("CK_REQUIRE_LLAMA_AVX512") && !ggml_cpu_has_avx512()) {
        std::fprintf(stderr,
                "Q6_K x Q8_K oracle requires an AVX-512 llama.cpp build, "
                "but ggml reports avx512=0\n");
        return 2;
    }
    std::printf("llama ISA: avx2=%d avx_vnni=%d avx512=%d avx512_vnni=%d; "
                "CK Q6 provider=%s\n",
            ggml_cpu_has_avx2(), ggml_cpu_has_avx_vnni(),
            ggml_cpu_has_avx512(), ggml_cpu_has_avx512_vnni(),
            ck_q6_k_q8_k_provider_name());
    if (!std::getenv("CK_NUM_THREADS")) setenv("CK_NUM_THREADS", "1", 1);
    const bool quick = argc > 1 && std::strcmp(argv[1], "--quick") == 0;
    const case_spec quick_cases[] = {
        {"decode_leaf_shape", 1, 64, 256},
        {"decode_practical", 1, 1024, 4096},
        {"short_prefill", 5, 512, 1024},
        {"multirow_prefill", 33, 512, 1024},
    };
    const case_spec full_cases[] = {
        {"decode_leaf_shape", 1, 64, 256},
        {"decode_practical", 1, 4096, 4096},
        {"short_prefill", 5, 1024, 4096},
        {"multirow_prefill", 33, 1024, 4096},
        {"long_k_prefill", 16, 512, 11008},
    };
    const case_spec *cases = quick ? quick_cases : full_cases;
    const size_t count = quick ? sizeof(quick_cases) / sizeof(quick_cases[0])
                               : sizeof(full_cases) / sizeof(full_cases[0]);
    int failed = 0;
    if (!run_q8_rounding_boundary_case()) ++failed;
    for (size_t i = 0; i < count; ++i) if (!run_case(cases[i])) ++failed;
    if (!run_xray_artifact_case()) ++failed;
    ck_threadpool_global_destroy();
    std::printf("\nQ6_K x Q8_K llama.cpp production parity: %s (%zu cases, %d failed)\n",
            failed ? "FAIL" : "PASS", count, failed);
    return failed ? 1 : 0;
}

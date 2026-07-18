// Authoritative RMSNorm + scale parity against the llama.cpp CPU graph.
//
// This test covers the numerical provider used by decoder RMSNorm and Q/K
// normalization.  In particular, it guards against changing the reciprocal
// square-root expression when a wider ISA is enabled.

#include "ggml.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

extern "C" {
void rmsnorm_forward_llama_production(
        const float * input, const float * gamma, float * output,
        float * rstd_cache, int tokens, int d_model, int aligned_embed_dim,
        float eps);
}

namespace {

struct case_spec {
    const char * name;
    int rows;
    int width;
    float eps;
};

static float fixture_value(int row, int col) {
    const float r = static_cast<float>(row);
    const float c = static_cast<float>(col);
    float value = 0.31f * std::sin(0.017f * c + 0.071f * r);
    value += 0.13f * std::cos(0.0031f * c - 0.019f * r);
    if ((row + col) % 127 == 0) {
        value += ((row + col) & 1) ? -0.9375f : 0.9375f;
    }
    return value;
}

static bool llama_rmsnorm_mul(
        const std::vector<float> & input,
        const std::vector<float> & gamma,
        std::vector<float> & output,
        int rows, int width, float eps) {
    const size_t arena_size = 16u * 1024u * 1024u
            + (input.size() + gamma.size() + output.size()) * sizeof(float);
    ggml_init_params params = {arena_size, nullptr, false};
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        return false;
    }

    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, width, rows);
    ggml_tensor * w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, width);
    std::memcpy(ggml_get_data(x), input.data(), input.size() * sizeof(float));
    std::memcpy(ggml_get_data(w), gamma.data(), gamma.size() * sizeof(float));

    ggml_tensor * norm = ggml_rms_norm(ctx, x, eps);
    ggml_tensor * scaled = ggml_mul(ctx, norm, w);
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, scaled);
    const int threads = std::max(1, std::atoi(
            std::getenv("CK_NUM_THREADS") ? std::getenv("CK_NUM_THREADS") : "1"));
    const bool ok = ggml_graph_compute_with_ctx(ctx, graph, threads) == GGML_STATUS_SUCCESS;
    if (ok) {
        std::memcpy(output.data(), ggml_get_data_f32(scaled), output.size() * sizeof(float));
    }
    ggml_free(ctx);
    return ok;
}

static bool run_case(const case_spec & spec) {
    const size_t count = static_cast<size_t>(spec.rows) * spec.width;
    std::vector<float> input(count);
    std::vector<float> gamma(spec.width);
    std::vector<float> ck(count, 0.0f);
    std::vector<float> llama(count, 0.0f);

    for (int row = 0; row < spec.rows; ++row) {
        for (int col = 0; col < spec.width; ++col) {
            input[static_cast<size_t>(row) * spec.width + col] = fixture_value(row, col);
        }
    }
    for (int col = 0; col < spec.width; ++col) {
        gamma[col] = 0.83f + 0.11f * std::sin(0.013f * static_cast<float>(col));
    }

    rmsnorm_forward_llama_production(
            input.data(), gamma.data(), ck.data(), nullptr,
            spec.rows, spec.width, spec.width, spec.eps);
    if (!llama_rmsnorm_mul(input, gamma, llama, spec.rows, spec.width, spec.eps)) {
        std::fprintf(stderr, "%s: llama.cpp graph execution failed\n", spec.name);
        return false;
    }

    size_t different = 0;
    size_t first = count;
    size_t worst = 0;
    float max_abs = 0.0f;
    for (size_t i = 0; i < count; ++i) {
        if (std::memcmp(&ck[i], &llama[i], sizeof(float)) != 0) {
            if (first == count) {
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
    if (different != 0) {
        std::printf(
                "%-24s different=%zu/%zu first=(%zu,%zu) worst=(%zu,%zu) "
                "max_abs=%.9g ck=%.9g llama=%.9g [FAIL]\n",
                spec.name, different, count,
                first / spec.width, first % spec.width,
                worst / spec.width, worst % spec.width,
                max_abs, ck[worst], llama[worst]);
        return false;
    }
    std::printf("%-24s bit_exact (%zu values) [PASS]\n", spec.name, count);
    return true;
}

} // namespace

int main() {
    const case_spec cases[] = {
        {"qk_norm_decode", 40, 128, 1.0e-6f},
        {"qk_norm_prefill", 40 * 17, 128, 1.0e-6f},
        {"decoder_rmsnorm", 4, 4096, 1.0e-6f},
    };
    int passed = 0;
    for (const case_spec & spec : cases) {
        passed += run_case(spec) ? 1 : 0;
    }
    std::printf("RMSNorm llama production: %d/%zu passed\n",
            passed, sizeof(cases) / sizeof(cases[0]));
    return passed == static_cast<int>(sizeof(cases) / sizeof(cases[0])) ? 0 : 1;
}

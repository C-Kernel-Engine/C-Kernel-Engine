// Authoritative recurrent Q/K L2 normalization parity against llama.cpp's CPU graph.

#include "ggml.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

extern "C" {
void recurrent_qk_l2_norm_forward(
        float * q, float * k, int rows, int q_dim, int k_dim,
        int head_dim, float eps);
}

namespace {

struct case_spec {
    const char * name;
    int rows;
    int q_dim;
    int k_dim;
    int head_dim;
    float eps;
    float input_scale;
};

static float fixture_value(int row, int col, int salt, float scale) {
    const float r = static_cast<float>(row + salt);
    const float c = static_cast<float>(col + 3 * salt);
    float value = 0.31f * std::sin(0.017f * c + 0.071f * r);
    value += 0.13f * std::cos(0.0031f * c - 0.019f * r);
    if ((row + col + salt) % 127 == 0) {
        value += ((row + col) & 1) ? -0.9375f : 0.9375f;
    }
    return scale * value;
}

static bool llama_l2_norm(
        const std::vector<float> & input,
        std::vector<float> & output,
        int rows, int dim, int head_dim, float eps) {
    const int heads = dim / head_dim;
    const int norm_rows = rows * heads;
    const size_t arena_size = 8u * 1024u * 1024u + 4u * input.size() * sizeof(float);
    ggml_init_params params = {arena_size, nullptr, false};
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        return false;
    }

    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, head_dim, norm_rows);
    std::memcpy(ggml_get_data(x), input.data(), input.size() * sizeof(float));
    ggml_tensor * normalized = ggml_l2_norm(ctx, x, eps);
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, normalized);
    const int threads = std::max(1, std::atoi(
            std::getenv("CK_NUM_THREADS") ? std::getenv("CK_NUM_THREADS") : "1"));
    const bool ok = ggml_graph_compute_with_ctx(ctx, graph, threads) == GGML_STATUS_SUCCESS;
    if (ok) {
        std::memcpy(output.data(), ggml_get_data_f32(normalized), output.size() * sizeof(float));
    }
    ggml_free(ctx);
    return ok;
}

static bool compare_exact(
        const char * name,
        const std::vector<float> & ck,
        const std::vector<float> & llama,
        int dim) {
    size_t different = 0;
    size_t first = ck.size();
    size_t worst = 0;
    float max_abs = 0.0f;
    for (size_t i = 0; i < ck.size(); ++i) {
        if (std::memcmp(&ck[i], &llama[i], sizeof(float)) != 0) {
            if (first == ck.size()) {
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
                "%-32s different=%zu/%zu first=(%zu,%zu) worst=(%zu,%zu) "
                "max_abs=%.9g ck=%.9g llama=%.9g [FAIL]\n",
                name, different, ck.size(), first / dim, first % dim,
                worst / dim, worst % dim, max_abs, ck[worst], llama[worst]);
        return false;
    }
    std::printf("%-32s bit_exact (%zu values) [PASS]\n", name, ck.size());
    return true;
}

static bool run_case(const case_spec & spec) {
    std::vector<float> q(static_cast<size_t>(spec.rows) * spec.q_dim);
    std::vector<float> k(static_cast<size_t>(spec.rows) * spec.k_dim);
    for (int row = 0; row < spec.rows; ++row) {
        for (int col = 0; col < spec.q_dim; ++col) {
            q[static_cast<size_t>(row) * spec.q_dim + col] =
                    fixture_value(row, col, 11, spec.input_scale);
        }
        for (int col = 0; col < spec.k_dim; ++col) {
            k[static_cast<size_t>(row) * spec.k_dim + col] =
                    fixture_value(row, col, 29, spec.input_scale);
        }
    }
    std::vector<float> ck_q = q;
    std::vector<float> ck_k = k;
    std::vector<float> llama_q(q.size());
    std::vector<float> llama_k(k.size());
    if (!llama_l2_norm(q, llama_q, spec.rows, spec.q_dim, spec.head_dim, spec.eps) ||
        !llama_l2_norm(k, llama_k, spec.rows, spec.k_dim, spec.head_dim, spec.eps)) {
        std::fprintf(stderr, "%s: llama.cpp graph execution failed\n", spec.name);
        return false;
    }
    recurrent_qk_l2_norm_forward(
            ck_q.data(), ck_k.data(), spec.rows, spec.q_dim, spec.k_dim,
            spec.head_dim, spec.eps);
    char q_name[96];
    char k_name[96];
    std::snprintf(q_name, sizeof(q_name), "%s.q", spec.name);
    std::snprintf(k_name, sizeof(k_name), "%s.k", spec.name);
    return compare_exact(q_name, ck_q, llama_q, spec.q_dim) &&
           compare_exact(k_name, ck_k, llama_k, spec.k_dim);
}

} // namespace

int main() {
    const case_spec cases[] = {
        {"small", 5, 64, 64, 16, 1.0e-6f, 1.0f},
        {"epsilon_clamp", 2, 32, 32, 8, 1.0e-6f, 1.0e-9f},
        {"qwen35_decode", 1, 2048, 2048, 128, 1.0e-6f, 1.0f},
        {"qwen35_prefill", 7, 2048, 2048, 128, 1.0e-6f, 1.0f},
        {"qwen35_parallel_prefill", 65, 2048, 2048, 128, 1.0e-6f, 1.0f},
    };
    int passed = 0;
    for (const case_spec & spec : cases) {
        passed += run_case(spec) ? 1 : 0;
    }
    std::printf("Recurrent Q/K L2 llama production: %d/%zu passed\n",
            passed, sizeof(cases) / sizeof(cases[0]));
    return passed == static_cast<int>(sizeof(cases) / sizeof(cases[0])) ? 0 : 1;
}

// Authoritative narrow FP32 projection parity against llama.cpp's CPU graph.

#include "ggml.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

extern "C" {
void gemm_nt_f32_llama_production(
        const float * A, const float * B, const float * bias, float * C,
        int M, int N, int K);
}

namespace {

struct case_spec {
    const char * name;
    int rows;
    int outputs;
    int width;
};

static float fixture(int row, int col, float phase) {
    float value = 0.31f * std::sin(
            0.017f * static_cast<float>(col)
            + 0.071f * static_cast<float>(row) + phase);
    value += 0.13f * std::cos(
            0.0031f * static_cast<float>(col)
            - 0.019f * static_cast<float>(row) - phase);
    if ((row + col) % 127 == 0) {
        value += ((row + col) & 1) ? -0.9375f : 0.9375f;
    }
    return value;
}

static bool llama_matmul(
        const std::vector<float> & input,
        const std::vector<float> & weight,
        std::vector<float> & output,
        const case_spec & spec) {
    const size_t arena_size = 16u * 1024u * 1024u
            + (input.size() + weight.size() + output.size()) * sizeof(float);
    ggml_init_params params = {arena_size, nullptr, false};
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        return false;
    }
    ggml_tensor * w = ggml_new_tensor_2d(
            ctx, GGML_TYPE_F32, spec.width, spec.outputs);
    ggml_tensor * x = ggml_new_tensor_2d(
            ctx, GGML_TYPE_F32, spec.width, spec.rows);
    std::memcpy(ggml_get_data(w), weight.data(), weight.size() * sizeof(float));
    std::memcpy(ggml_get_data(x), input.data(), input.size() * sizeof(float));
    ggml_tensor * y = ggml_mul_mat(ctx, w, x);
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, y);
    const int threads = std::max(1, std::atoi(
            std::getenv("CK_NUM_THREADS") ? std::getenv("CK_NUM_THREADS") : "1"));
    const bool ok =
            ggml_graph_compute_with_ctx(ctx, graph, threads) == GGML_STATUS_SUCCESS;
    if (ok) {
        std::memcpy(output.data(), ggml_get_data_f32(y),
                output.size() * sizeof(float));
    }
    ggml_free(ctx);
    return ok;
}

static bool run_case(const case_spec & spec) {
    std::vector<float> input(
            static_cast<size_t>(spec.rows) * spec.width);
    std::vector<float> weight(
            static_cast<size_t>(spec.outputs) * spec.width);
    std::vector<float> ck(
            static_cast<size_t>(spec.rows) * spec.outputs, 0.0f);
    std::vector<float> llama(ck.size(), 0.0f);
    for (int row = 0; row < spec.rows; ++row) {
        for (int col = 0; col < spec.width; ++col) {
            input[static_cast<size_t>(row) * spec.width + col] =
                    fixture(row, col, 0.17f);
        }
    }
    for (int row = 0; row < spec.outputs; ++row) {
        for (int col = 0; col < spec.width; ++col) {
            weight[static_cast<size_t>(row) * spec.width + col] =
                    fixture(row, col, -0.23f) * 0.071f;
        }
    }
    gemm_nt_f32_llama_production(
            input.data(), weight.data(), nullptr, ck.data(),
            spec.rows, spec.outputs, spec.width);
    if (!llama_matmul(input, weight, llama, spec)) {
        std::fprintf(stderr, "%s: llama.cpp graph execution failed\n", spec.name);
        return false;
    }
    size_t different = 0;
    float max_abs = 0.0f;
    for (size_t i = 0; i < ck.size(); ++i) {
        different += std::memcmp(&ck[i], &llama[i], sizeof(float)) != 0;
        max_abs = std::max(max_abs, std::fabs(ck[i] - llama[i]));
    }
    std::printf("%-24s different=%zu/%zu max_abs=%.9g [%s]\n",
            spec.name, different, ck.size(), max_abs,
            different == 0 ? "PASS" : "FAIL");
    return different == 0;
}

} // namespace

int main() {
    const case_spec cases[] = {
        {"decode_narrow", 1, 48, 5120},
        {"prefill_four", 4, 48, 5120},
        {"prefill_chunk_tail", 65, 48, 5120},
        {"qwen35_router_chunk", 65, 256, 2048},
    };
    int passed = 0;
    for (const case_spec & spec : cases) {
        passed += run_case(spec) ? 1 : 0;
    }
    std::printf("FP32 GEMM llama production: %d/%zu passed\n",
            passed, sizeof(cases) / sizeof(cases[0]));
    return passed == static_cast<int>(sizeof(cases) / sizeof(cases[0])) ? 0 : 1;
}

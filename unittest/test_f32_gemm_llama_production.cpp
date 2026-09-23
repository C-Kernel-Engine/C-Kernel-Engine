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
void gemm_nt_f32_llama_production_output_range(
        const float * A, const float * B, const float * bias, float * C,
        int M, int N, int K, int output_begin, int output_end);
void gemm_nt_f32_llama_production_parallel_dispatch(
        const float * A, const float * B, const float * bias, float * C,
        int M, int N, int K);
void ck_f32_gemm_profile_reset(void);
size_t ck_f32_gemm_profile_count(void);
uint64_t ck_f32_gemm_profile_overflow_calls(void);
int ck_f32_gemm_profile_get(
        size_t index, int * M, int * N, int * K, int * active_threads,
        int * parallel, uint64_t * calls, uint64_t * elapsed_ns);
}

namespace {

struct case_spec {
    const char * name;
    int rows;
    int outputs;
    int width;
    bool with_bias = false;
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
    std::vector<float> ck_ranges(ck.size(), 0.0f);
    std::vector<float> ck_parallel(ck.size(), 0.0f);
    std::vector<float> llama(ck.size(), 0.0f);
    std::vector<float> bias(spec.outputs, 0.0f);
    for (int row = 0; row < spec.rows; ++row) {
        for (int col = 0; col < spec.width; ++col) {
            input[static_cast<size_t>(row) * spec.width + col] =
                    fixture(row, col, 0.17f);
        }
    }
    for (int row = 0; row < spec.outputs; ++row) {
        bias[row] = fixture(row, 0, 0.41f) * 0.019f;
        for (int col = 0; col < spec.width; ++col) {
            weight[static_cast<size_t>(row) * spec.width + col] =
                    fixture(row, col, -0.23f) * 0.071f;
        }
    }
    const float * bias_data = spec.with_bias ? bias.data() : nullptr;
    gemm_nt_f32_llama_production(
            input.data(), weight.data(), bias_data, ck.data(),
            spec.rows, spec.outputs, spec.width);
    gemm_nt_f32_llama_production_parallel_dispatch(
            input.data(), weight.data(), bias_data, ck_parallel.data(),
            spec.rows, spec.outputs, spec.width);
    const int total = spec.rows * spec.outputs;
    const int chunk = std::max(1, (total + 6) / 7);
    for (int begin = 0; begin < total; begin += chunk) {
        gemm_nt_f32_llama_production_output_range(
                input.data(), weight.data(), bias_data, ck_ranges.data(),
                spec.rows, spec.outputs, spec.width,
                begin, std::min(begin + chunk, total));
    }
    for (size_t i = 0; i < ck.size(); ++i) {
        if (std::memcmp(&ck[i], &ck_ranges[i], sizeof(float)) != 0) {
            std::fprintf(stderr, "%s: output-range mismatch at %zu\n", spec.name, i);
            return false;
        }
        if (std::memcmp(&ck[i], &ck_parallel[i], sizeof(float)) != 0) {
            std::fprintf(stderr, "%s: parallel-dispatch mismatch at %zu\n", spec.name, i);
            return false;
        }
    }
    if (spec.with_bias) {
        std::printf("%-24s internal_serial_range_parallel=bit_exact [PASS]\n",
                spec.name);
        return true;
    }
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
        {"whisper_decode_hidden", 1, 512, 512},
        {"whisper_decode_mlp_up", 1, 2048, 512},
        {"whisper_decode_mlp_down", 1, 512, 2048},
        {"whisper_decode_logits", 1, 51865, 512},
        {"decode_narrow", 1, 48, 5120},
        {"prefill_four", 4, 48, 5120},
        {"prefill_chunk_tail", 65, 48, 5120},
        {"qwen35_router_chunk", 65, 256, 2048},
        {"prefill_output_tail_bias", 7, 127, 128, true},
        {"prefill_reduction_tail_bias", 7, 127, 130, true},
        {"prefill_partition_tail_bias", 9, 131, 512, true},
        {"audio_ffn_up", 9, 5120, 1280},
        {"audio_ffn_down", 9, 1280, 5120},
    };
    int passed = 0;
    ck_f32_gemm_profile_reset();
    for (const case_spec & spec : cases) {
        passed += run_case(spec) ? 1 : 0;
    }
    const size_t case_count = sizeof(cases) / sizeof(cases[0]);
    bool profile_ok = ck_f32_gemm_profile_count() == case_count
            && ck_f32_gemm_profile_overflow_calls() == 0;
    for (size_t index = 0; profile_ok && index < case_count; ++index) {
        int M = 0, N = 0, K = 0, active_threads = 0, parallel = -1;
        uint64_t calls = 0, elapsed_ns = 0;
        profile_ok = ck_f32_gemm_profile_get(
                index, &M, &N, &K, &active_threads, &parallel,
                &calls, &elapsed_ns) == 0
                && M == cases[index].rows
                && N == cases[index].outputs
                && K == cases[index].width
                && active_threads >= 1
                && (parallel == 0 || parallel == 1)
                && calls == 1
                && elapsed_ns > 0;
    }
    int unused = 0;
    uint64_t unused_u64 = 0;
    profile_ok = profile_ok && ck_f32_gemm_profile_get(
            case_count, &unused, &unused, &unused, &unused, &unused,
            &unused_u64, &unused_u64) != 0;
    std::printf("FP32 GEMM profile: %zu/%zu shapes [%s]\n",
            ck_f32_gemm_profile_count(), case_count,
            profile_ok ? "PASS" : "FAIL");
    std::printf("FP32 GEMM llama production: %d/%zu passed\n",
            passed, case_count);
    return passed == static_cast<int>(case_count) && profile_ok ? 0 : 1;
}

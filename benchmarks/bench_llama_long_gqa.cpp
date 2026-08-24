#include "ggml-cpu.h"
#include "ggml.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>

static size_t tensor_offset(
    const ggml_tensor *tensor, int i0, int i1, int i2, int i3)
{
    return (size_t)i0 * tensor->nb[0] + (size_t)i1 * tensor->nb[1] +
           (size_t)i2 * tensor->nb[2] + (size_t)i3 * tensor->nb[3];
}

static void fill_inputs(
    ggml_tensor *q, ggml_tensor *k, ggml_tensor *v,
    int num_heads, int num_kv_heads, int query_tokens,
    int kv_tokens, int head_dim)
{
    for (int h = 0; h < num_heads; ++h) {
        for (int tq = 0; tq < query_tokens; ++tq) {
            for (int d = 0; d < head_dim; ++d) {
                const float value = ((float)d - 0.5f * (float)(head_dim - 1)) /
                                    (float)(2 * head_dim);
                memcpy((char *)q->data + tensor_offset(q, d, tq, h, 0),
                       &value, sizeof(value));
            }
        }
    }
    for (int h = 0; h < num_kv_heads; ++h) {
        for (int tk = 0; tk < kv_tokens; ++tk) {
            for (int d = 0; d < head_dim; ++d) {
                const float k_value = ((float)d - 0.5f * (float)(head_dim - 1)) /
                                      (float)head_dim;
                const float v_value = (0.375f * (float)(head_dim - 1 - 2 * d)) /
                                      (float)(head_dim - 1);
                const ggml_fp16_t k_half = ggml_fp32_to_fp16(k_value);
                const ggml_fp16_t v_half = ggml_fp32_to_fp16(v_value);
                memcpy((char *)k->data + tensor_offset(k, d, tk, h, 0),
                       &k_half, sizeof(k_half));
                memcpy((char *)v->data + tensor_offset(v, d, tk, h, 0),
                       &v_half, sizeof(v_half));
            }
        }
    }
}

static void fill_causal_mask(
    ggml_tensor *mask, int query_tokens, int kv_tokens, int past_tokens)
{
    memset(mask->data, 0, ggml_nbytes(mask));
    const ggml_fp16_t negative_infinity =
        ggml_fp32_to_fp16(-std::numeric_limits<float>::infinity());
    for (int tq = 0; tq < query_tokens; ++tq) {
        const int first_masked = past_tokens + tq + 1;
        for (int tk = first_masked; tk < kv_tokens; ++tk) {
            memcpy((char *)mask->data + tensor_offset(mask, tk, tq, 0, 0),
                   &negative_infinity, sizeof(negative_infinity));
        }
    }
}

static uint64_t output_hash(const ggml_tensor *output)
{
    const uint8_t *bytes = (const uint8_t *)output->data;
    const size_t count = ggml_nbytes(output);
    uint64_t hash = UINT64_C(1469598103934665603);
    for (size_t i = 0; i < count; ++i) {
        hash ^= bytes[i];
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

int main(int argc, char **argv)
{
    if (argc != 9) {
        fprintf(stderr,
                "usage: %s H Hkv Q KV D threads warmups iterations\n",
                argv[0]);
        return 2;
    }
    const int num_heads = atoi(argv[1]);
    const int num_kv_heads = atoi(argv[2]);
    const int query_tokens = atoi(argv[3]);
    const int kv_tokens = atoi(argv[4]);
    const int head_dim = atoi(argv[5]);
    const int threads = atoi(argv[6]);
    const int warmups = atoi(argv[7]);
    const int iterations = atoi(argv[8]);
    if (num_heads <= 0 || num_kv_heads <= 0 ||
        num_heads % num_kv_heads != 0 || query_tokens <= 0 ||
        kv_tokens < query_tokens || head_dim <= 0 || threads <= 0 ||
        warmups < 0 || iterations <= 0) {
        return 2;
    }

    ggml_cpu_init();
    const size_t q_count = (size_t)num_heads * (size_t)query_tokens *
                           (size_t)head_dim;
    const size_t kv_count = (size_t)num_kv_heads * (size_t)kv_tokens *
                            (size_t)head_dim;
    const size_t mask_count = (size_t)query_tokens * (size_t)kv_tokens;
    const size_t memory = 256u * 1024u * 1024u +
                          2u * q_count * sizeof(float) +
                          2u * kv_count * sizeof(ggml_fp16_t) +
                          mask_count * sizeof(ggml_fp16_t);
    const ggml_init_params params = {memory, nullptr, false};
    ggml_context *context = ggml_init(params);
    if (!context) {
        fprintf(stderr, "ggml_init failed for %zu bytes\n", memory);
        return 3;
    }

    ggml_tensor *q = ggml_new_tensor_4d(
        context, GGML_TYPE_F32, head_dim, query_tokens, num_heads, 1);
    ggml_tensor *k = ggml_new_tensor_4d(
        context, GGML_TYPE_F16, head_dim, kv_tokens, num_kv_heads, 1);
    ggml_tensor *v = ggml_new_tensor_4d(
        context, GGML_TYPE_F16, head_dim, kv_tokens, num_kv_heads, 1);
    ggml_tensor *mask = ggml_new_tensor_2d(
        context, GGML_TYPE_F16, kv_tokens, query_tokens);
    fill_inputs(q, k, v, num_heads, num_kv_heads, query_tokens,
                kv_tokens, head_dim);
    fill_causal_mask(mask, query_tokens, kv_tokens, kv_tokens - query_tokens);

    ggml_tensor *output = ggml_flash_attn_ext(
        context, q, k, v, mask, 1.0f / sqrtf((float)head_dim), 0.0f, 0.0f);
    ggml_flash_attn_ext_set_prec(output, GGML_PREC_F32);
    ggml_cgraph *graph = ggml_new_graph(context);
    ggml_build_forward_expand(graph, output);
    ggml_threadpool_params threadpool_params =
        ggml_threadpool_params_default(threads);
    threadpool_params.paused = false;
    ggml_threadpool *threadpool = ggml_threadpool_new(&threadpool_params);
    ggml_cplan plan = ggml_graph_plan(graph, threads, threadpool);
    plan.work_data = plan.work_size ? (uint8_t *)malloc(plan.work_size) : nullptr;
    if (plan.work_size && !plan.work_data) {
        fprintf(stderr, "work allocation failed for %zu bytes\n", plan.work_size);
        return 4;
    }

    for (int i = 0; i < warmups; ++i) {
        if (ggml_graph_compute(graph, &plan) != GGML_STATUS_SUCCESS) return 5;
    }
    const auto started = std::chrono::steady_clock::now();
    for (int i = 0; i < iterations; ++i) {
        if (ggml_graph_compute(graph, &plan) != GGML_STATUS_SUCCESS) return 5;
    }
    const auto finished = std::chrono::steady_clock::now();
    const double seconds =
        std::chrono::duration<double>(finished - started).count() /
        (double)iterations;
    const uint64_t hash = output_hash(output);

    printf("{\"runtime\":\"llama.cpp\",\"heads\":%d,\"kv_heads\":%d,"
           "\"query_tokens\":%d,\"context_tokens\":%d,\"head_dim\":%d,"
           "\"threads\":%d,\"seconds\":%.9f,\"output_fnv1a64\":\"%016llx\"}\n",
           num_heads, num_kv_heads, query_tokens, kv_tokens, head_dim,
           threads, seconds, (unsigned long long)hash);

    free(plan.work_data);
    ggml_threadpool_free(threadpool);
    ggml_free(context);
    return 0;
}

#include "ggml.h"
#include "ggml-cpu.h"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

extern "C" void geglu_forward_ggml_native(const float *, float *, int, int);

int main() {
    for (int rows : {1, 26}) {
        const int dim = 2048;
        std::vector<float> input((size_t)rows * dim * 2);
        for (size_t i = 0; i < input.size(); ++i)
            input[i] = 12.0f * std::sin((float)i * 0.017f);
        input[0] = -10.0f; input[1] = 10.0f; input[2] = -0.0f;
        ggml_context *ctx = ggml_init({32u * 1024u * 1024u, nullptr, false});
        if (!ctx) return 2;
        ggml_tensor *x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2 * dim, rows);
        std::memcpy(ggml_get_data(x), input.data(), input.size() * sizeof(float));
        ggml_tensor *y = ggml_geglu(ctx, x);
        ggml_cgraph *graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, y);
        if (ggml_graph_compute_with_ctx(ctx, graph, 4) != GGML_STATUS_SUCCESS) return 2;
        for (bool inplace : {false, true}) {
            std::vector<float> actual(inplace ? input : std::vector<float>((size_t)rows * dim));
            geglu_forward_ggml_native(inplace ? actual.data() : input.data(), actual.data(), rows, dim);
            if (std::memcmp(actual.data(), ggml_get_data(y), (size_t)rows * dim * sizeof(float))) {
                std::fprintf(stderr, "FAIL rows=%d inplace=%d\n", rows, inplace);
                ggml_free(ctx);
                return 1;
            }
            std::printf("PASS bit-exact rows=%d inplace=%d\n", rows, inplace);
        }
        ggml_free(ctx);
    }
    return 0;
}

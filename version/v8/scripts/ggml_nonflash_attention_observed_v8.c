#include "ggml.h"
#include "ggml-cpu.h"

#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int parse_size(const char *text, size_t *value) {
    char *end = NULL;
    errno = 0;
    const unsigned long long parsed = strtoull(text, &end, 10);
    if (errno || end == text || *end || !parsed || parsed > 64u * 1024u * 1024u) return 0;
    *value = (size_t) parsed;
    return 1;
}

static int parse_float(const char *text, float *value) {
    char *end = NULL;
    errno = 0;
    const float parsed = strtof(text, &end);
    if (errno || end == text || *end || !isfinite(parsed)) return 0;
    *value = parsed;
    return 1;
}

static void *read_exact(const char *path, size_t bytes) {
    FILE *file = fopen(path, "rb");
    void *buffer = malloc(bytes);
    if (!file || !buffer || fread(buffer, 1, bytes, file) != bytes || fgetc(file) != EOF) {
        if (file) fclose(file);
        free(buffer);
        return NULL;
    }
    fclose(file);
    return buffer;
}

int main(int argc, char **argv) {
    if (argc != 21) {
        fprintf(stderr, "usage: observed Q K V MASK OUT queries keys heads kv_heads dim "
                "q_nb1 q_nb2 k_nb1 k_nb2 v_nb1 v_nb2 mask_nb1 scale alibi threads\n");
        return 2;
    }
    size_t n[5], stride[7], threads;
    float scale, alibi;
    for (int i = 0; i < 5; ++i) if (!parse_size(argv[6 + i], &n[i])) return 2;
    for (int i = 0; i < 7; ++i) if (!parse_size(argv[11 + i], &stride[i])) return 2;
    if (!parse_float(argv[18], &scale) || scale <= 0 ||
        !parse_float(argv[19], &alibi) || !parse_size(argv[20], &threads)) return 2;
    const size_t queries = n[0], keys = n[1], heads = n[2], kv_heads = n[3], dim = n[4];
    if (queries > 4096 || keys > 4096 || heads > 128 || kv_heads > heads ||
        heads % kv_heads || dim > 1024 || threads > 256 ||
        stride[0] < dim * 4 || stride[2] < dim * 2 || stride[4] < keys * 2 ||
        stride[6] < keys * 4) return 2;
    const size_t qbytes = (heads - 1) * stride[1] + (queries - 1) * stride[0] + dim * 4;
    const size_t kbytes = (kv_heads - 1) * stride[3] + (keys - 1) * stride[2] + dim * 2;
    const size_t vbytes = (kv_heads - 1) * stride[5] + (dim - 1) * stride[4] + keys * 2;
    const size_t mbytes = (queries - 1) * stride[6] + keys * 4;
    if (qbytes > 64u * 1024u * 1024u || kbytes > 64u * 1024u * 1024u ||
        vbytes > 64u * 1024u * 1024u || mbytes > 64u * 1024u * 1024u ||
        qbytes + kbytes + vbytes + mbytes > 64u * 1024u * 1024u ||
        queries * keys * heads > 4u * 1024u * 1024u ||
        queries * heads * dim * sizeof(float) > 64u * 1024u * 1024u) return 2;
    void *raw[4] = {read_exact(argv[1], qbytes), read_exact(argv[2], kbytes),
                    read_exact(argv[3], vbytes), read_exact(argv[4], mbytes)};
    if (!raw[0] || !raw[1] || !raw[2] || !raw[3]) {
        fprintf(stderr, "missing or incorrectly sized observed operand\n");
        for (int i = 0; i < 4; ++i) free(raw[i]);
        return 2;
    }

    ggml_cpu_init();
    const size_t context_bytes = 32u * 1024u * 1024u +
        2 * (qbytes + kbytes + vbytes + mbytes) +
        3 * (queries * keys * heads * sizeof(float)) +
        2 * (queries * heads * dim * sizeof(float));
    if (context_bytes > 256u * 1024u * 1024u) return 2;
    struct ggml_init_params params = {.mem_size = context_bytes, .mem_buffer = NULL,
                                      .no_alloc = false};
    struct ggml_context *ctx = ggml_init(params);
    if (!ctx) return 3;
    struct ggml_tensor *q = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, dim, queries, heads);
    struct ggml_tensor *k = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, dim, keys, kv_heads);
    struct ggml_tensor *v = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, keys, dim, kv_heads);
    struct ggml_tensor *mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, keys, queries);
    if (!q || !k || !v || !mask) return 3;
    q->data = raw[0]; q->nb[1] = stride[0]; q->nb[2] = stride[1];
    k->data = raw[1]; k->nb[1] = stride[2]; k->nb[2] = stride[3];
    v->data = raw[2]; v->nb[1] = stride[4]; v->nb[2] = stride[5];
    mask->data = raw[3]; mask->nb[1] = stride[6];
    struct ggml_tensor *scores = ggml_mul_mat(ctx, k, q);
    ggml_prec_set_acc(scores, GGML_PREC_F32);
    struct ggml_tensor *prob = ggml_soft_max_ext(ctx, scores, mask, scale, alibi);
    struct ggml_tensor *value = ggml_mul_mat(ctx, v, prob);
    struct ggml_tensor *result = ggml_permute(ctx, value, 0, 2, 1, 3);
    result = ggml_cont_2d(ctx, result, dim * heads, queries);
    struct ggml_cgraph *graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, result);
    const int computed = ggml_graph_compute_with_ctx(ctx, graph, (int) threads) == GGML_STATUS_SUCCESS;
    FILE *file = computed ? fopen(argv[5], "wb") : NULL;
    int ok = file && fwrite(result->data, sizeof(float), queries * heads * dim, file) ==
                     queries * heads * dim;
    if (file && fclose(file) != 0) ok = 0;
    for (int i = 0; i < 4; ++i) free(raw[i]);
    ggml_free(ctx);
    if (!ok) fprintf(stderr, "observed attention graph failed or produced incomplete output\n");
    return ok ? 0 : 3;
}

#ifndef CK_GGML_RUNTIME_COMPAT_H
#define CK_GGML_RUNTIME_COMPAT_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

typedef uint16_t ggml_fp16_t;

struct ggml_context;
struct ggml_tensor;
struct ggml_cgraph;

struct ggml_init_params {
    size_t mem_size;
    void * mem_buffer;
    bool no_alloc;
};

enum ggml_type {
    GGML_TYPE_F32 = 0,
    GGML_TYPE_F16 = 1,
    GGML_TYPE_Q8_0 = 8,
    GGML_TYPE_I32 = 26,
};

enum ggml_status {
    GGML_STATUS_ALLOC_FAILED = -2,
    GGML_STATUS_FAILED = -1,
    GGML_STATUS_SUCCESS = 0,
    GGML_STATUS_ABORTED = 1,
};

#ifndef GGML_MROPE_SECTIONS
#define GGML_MROPE_SECTIONS 4
#endif

#ifndef GGML_ROPE_TYPE_VISION
#define GGML_ROPE_TYPE_VISION 24
#endif

float ggml_fp16_to_fp32(ggml_fp16_t value);
ggml_fp16_t ggml_fp32_to_fp16(float value);
void *ggml_get_data(const struct ggml_tensor *tensor);
float *ggml_get_data_f32(const struct ggml_tensor *tensor);
size_t ggml_nbytes(const struct ggml_tensor *tensor);

#endif

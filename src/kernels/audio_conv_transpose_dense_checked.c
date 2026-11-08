#include "ckernel_audio_conv_transpose_dense.h"
#include "checked_float_buffer.h"

#include <math.h>
#include <stdint.h>
#include <string.h>

#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
#define CK_DECONV_AVX2 1
/* Channels are independent. Within each channel retain input-channel then
 * input-frame order; SIMD tap lanes update distinct output positions. */
__attribute__((target("avx2,fma"), noinline))
static void dense_deconv_compute_avx2(const float *input, size_t input_stride,
    const float *weight, float *scratch, size_t input_channels,
    size_t output_channels, size_t input_frames, size_t kernel_size,
    size_t stride, size_t padding, size_t output_frames) {
    const size_t limit = padding + output_frames;
    for (size_t oc = 0; oc < output_channels; ++oc) {
        float *out = scratch + oc * output_frames;
        for (size_t ic = 0; ic < input_channels; ++ic) {
            const float *w = weight + (ic * output_channels + oc) * kernel_size;
            for (size_t frame = 0; frame < input_frames; ++frame) {
                const size_t base = frame * stride;
                if (base >= limit) continue;
                size_t tap = base < padding ? padding - base : 0;
                size_t end = limit - base;
                if (end > kernel_size) end = kernel_size;
                const float value = input[ic * input_stride + frame];
                const __m256 x = _mm256_set1_ps(value);
                for (; tap < end && end - tap >= 8; tap += 8) {
                    float *dest = out + (base + tap - padding);
                    _mm256_storeu_ps(dest, _mm256_fmadd_ps(x,
                        _mm256_loadu_ps(w + tap), _mm256_loadu_ps(dest)));
                }
                const __m128 x4 = _mm_set1_ps(value);
                for (; tap < end && end - tap >= 4; tap += 4) {
                    float *dest = out + (base + tap - padding);
                    _mm_storeu_ps(dest, _mm_fmadd_ps(x4,
                        _mm_loadu_ps(w + tap), _mm_loadu_ps(dest)));
                }
                for (; tap < end; ++tap) {
                    const size_t pos = base + tap - padding;
                    out[pos] = fmaf(value, w[tap], out[pos]);
                }
            }
        }
    }
}
#else
#define CK_DECONV_AVX2 0
#endif

int audio_conv_transpose1d_dense_f32_workspace(
    size_t output_channels, size_t output_frames, size_t *scratch_elements) {
    if (!scratch_elements || !output_channels || !output_frames ||
        output_channels > SIZE_MAX / output_frames ||
        output_channels * output_frames > SIZE_MAX / sizeof(float))
        return CK_AUDIO_DECONV_DENSE_OVERFLOW;
    *scratch_elements = output_channels * output_frames;
    return CK_AUDIO_DECONV_DENSE_OK;
}

static int dense_deconv_checked(
    const float *input, size_t input_elements, size_t input_stride,
    const float *weight, size_t weight_elements,
    const float *bias, size_t bias_elements,
    float *output, size_t output_elements, size_t output_stride,
    float *scratch, size_t scratch_elements,
    size_t input_channels, size_t output_channels, size_t input_frames,
    size_t kernel_size, size_t stride, size_t padding,
    size_t output_padding, size_t right_crop, size_t output_frames) {
    size_t geometry, input_span, output_span, weight_need, scratch_need;
    if (!input_channels || !output_channels || !input_frames ||
        !kernel_size || !stride || !output_frames || output_padding >= stride)
        return CK_AUDIO_DECONV_DENSE_INVALID;
    if (input_frames - 1 > SIZE_MAX / stride || padding > SIZE_MAX / 2)
        return CK_AUDIO_DECONV_DENSE_OVERFLOW;
    geometry = (input_frames - 1) * stride;
    if (geometry > SIZE_MAX - kernel_size ||
        geometry + kernel_size > SIZE_MAX - output_padding)
        return CK_AUDIO_DECONV_DENSE_OVERFLOW;
    geometry += kernel_size + output_padding;
    if (2 * padding >= geometry || right_crop >= geometry - 2 * padding ||
        geometry - 2 * padding - right_crop != output_frames)
        return CK_AUDIO_DECONV_DENSE_INVALID;
    if (!ck_checked_span(input_channels, input_frames, input_stride,
                         &input_span) ||
        !ck_checked_span(output_channels, output_frames, output_stride,
                         &output_span) ||
        input_channels > SIZE_MAX / output_channels ||
        input_channels * output_channels > SIZE_MAX / kernel_size ||
        audio_conv_transpose1d_dense_f32_workspace(
            output_channels, output_frames, &scratch_need))
        return CK_AUDIO_DECONV_DENSE_OVERFLOW;
    weight_need = input_channels * output_channels * kernel_size;
    if (weight_need > SIZE_MAX / sizeof(float))
        return CK_AUDIO_DECONV_DENSE_OVERFLOW;
    if (input_elements < input_span || weight_elements < weight_need ||
        bias_elements < output_channels || output_elements < output_span ||
        scratch_elements < scratch_need)
        return CK_AUDIO_DECONV_DENSE_CAPACITY;
    if (!ck_checked_region(input, input_span) ||
        !ck_checked_region(weight, weight_need) ||
        !ck_checked_region(bias, output_channels) ||
        !ck_checked_region(output, output_span) ||
        !ck_checked_region(scratch, scratch_need) ||
        ck_checked_overlap(output, output_span, input, input_span) ||
        ck_checked_overlap(output, output_span, weight, weight_need) ||
        ck_checked_overlap(output, output_span, bias, output_channels) ||
        ck_checked_overlap(output, output_span, scratch, scratch_need) ||
        ck_checked_overlap(scratch, scratch_need, input, input_span) ||
        ck_checked_overlap(scratch, scratch_need, weight, weight_need) ||
        ck_checked_overlap(scratch, scratch_need, bias, output_channels))
        return CK_AUDIO_DECONV_DENSE_INVALID;
    if (!ck_checked_finite(weight, weight_need) ||
        !ck_checked_finite(bias, output_channels))
        return CK_AUDIO_DECONV_DENSE_NONFINITE;
    for (size_t ic = 0; ic < input_channels; ++ic)
        if (!ck_checked_finite(input + ic * input_stride, input_frames))
            return CK_AUDIO_DECONV_DENSE_NONFINITE;

    for (size_t oc = 0; oc < output_channels; ++oc)
        for (size_t frame = 0; frame < output_frames; ++frame)
            scratch[oc * output_frames + frame] = bias[oc];
#if CK_DECONV_AVX2
    __builtin_cpu_init();
    if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma")) {
        dense_deconv_compute_avx2(input, input_stride, weight, scratch,
            input_channels, output_channels, input_frames, kernel_size,
            stride, padding, output_frames);
    } else
#endif
    {
        for (size_t ic = 0; ic < input_channels; ++ic)
            for (size_t frame = 0; frame < input_frames; ++frame)
                for (size_t tap = 0; tap < kernel_size; ++tap) {
                    size_t target = frame * stride + tap;
                    if (target < padding || target - padding >= output_frames)
                        continue;
                    size_t out_frame = target - padding;
                    float value = input[ic * input_stride + frame];
                    for (size_t oc = 0; oc < output_channels; ++oc) {
                        size_t index = oc * output_frames + out_frame;
                        scratch[index] = fmaf(value,
                            weight[(ic * output_channels + oc) * kernel_size + tap],
                            scratch[index]);
                    }
                }
    }
    if (!ck_checked_finite(scratch, scratch_need))
        return CK_AUDIO_DECONV_DENSE_NONFINITE;
    for (size_t oc = 0; oc < output_channels; ++oc)
        memcpy(output + oc * output_stride,
               scratch + oc * output_frames,
               output_frames * sizeof(float));
    return CK_AUDIO_DECONV_DENSE_OK;
}

int audio_conv_transpose1d_dense_channel_major_f32_checked(
    const float *input, size_t input_elements, size_t input_stride,
    const float *weight, size_t weight_elements,
    const float *bias, size_t bias_elements,
    float *output, size_t output_elements, size_t output_stride,
    float *scratch, size_t scratch_elements,
    size_t input_channels, size_t output_channels, size_t input_frames,
    size_t kernel_size, size_t stride, size_t padding,
    size_t output_padding, size_t output_frames) {
    return dense_deconv_checked(input, input_elements, input_stride,
        weight, weight_elements, bias, bias_elements, output,
        output_elements, output_stride, scratch, scratch_elements,
        input_channels, output_channels, input_frames, kernel_size,
        stride, padding, output_padding, 0, output_frames);
}

int audio_conv_transpose1d_dense_crop_right_channel_major_f32_checked(
    const float *input, size_t input_elements, size_t input_stride,
    const float *weight, size_t weight_elements,
    const float *bias, size_t bias_elements,
    float *output, size_t output_elements, size_t output_stride,
    float *scratch, size_t scratch_elements,
    size_t input_channels, size_t output_channels, size_t input_frames,
    size_t kernel_size, size_t stride, size_t padding,
    size_t output_padding, size_t right_crop, size_t output_frames) {
    return dense_deconv_checked(input, input_elements, input_stride,
        weight, weight_elements, bias, bias_elements, output,
        output_elements, output_stride, scratch, scratch_elements,
        input_channels, output_channels, input_frames, kernel_size,
        stride, padding, output_padding, right_crop, output_frames);
}

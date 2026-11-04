#include "ckernel_audio_conv_checked.h"
#include "checked_float_buffer.h"
#include "audio_conv1d_checked_internal.h"

#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

#if (defined(__x86_64__) || defined(__i386__)) && \
    (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
#endif

#if (defined(__x86_64__) || defined(__i386__)) && \
    (defined(__GNUC__) || defined(__clang__))
#define CK_AUDIO_CONV_X86_TARGET_FMA 1
#else
#define CK_AUDIO_CONV_X86_TARGET_FMA 0
#endif

/* Both implementations execute the same channel/frame/input/tap reduction.
 * Only the function target changes how the explicit fmaf is implemented. */
#define CK_AUDIO_CONV1D_COMPUTE_BODY \
    for (size_t oc = job->channel_begin; oc < job->channel_end; ++oc) { \
        for (size_t frame = 0; frame < job->output_frames; ++frame) { \
            float sum = job->bias[oc]; \
            for (size_t ic = 0; ic < job->input_channels; ++ic) { \
                const float *row = job->weight + \
                    (oc * job->input_channels + ic) * job->kernel_size; \
                for (size_t tap = 0; tap < job->kernel_size; ++tap) { \
                    int64_t source = (int64_t)frame * job->stride + \
                        (int64_t)tap * job->dilation - job->padding; \
                    if (source >= 0 && (size_t)source < job->input_frames) \
                        sum = fmaf(job->input[ic * job->input_stride + \
                            (size_t)source], row[tap], sum); \
                } \
            } \
            job->staged[oc * job->output_frames + frame] = sum; \
        } \
    }

static void audio_conv1d_compute_scalar(const ck_audio_conv1d_job *job, void *unused) {
    (void)unused;
    CK_AUDIO_CONV1D_COMPUTE_BODY
}

#if CK_AUDIO_CONV_X86_TARGET_FMA
__attribute__((target("fma"), noinline))
static void audio_conv1d_compute_hwfma(const ck_audio_conv1d_job *job, void *unused) {
    (void)unused;
    CK_AUDIO_CONV1D_COMPUTE_BODY
}

/* Output frames are independent. SIMD lanes retain the input-channel/tap
 * reduction order of the scalar FMA provider; padding frames use its exact
 * scalar path, so no zero-padded arithmetic is introduced. */
__attribute__((target("avx2,fma"), noinline))
void audio_conv1d_compute_avx2(const ck_audio_conv1d_job *job, void *unused) {
    (void)unused;
    if (job->stride != 1 || job->output_frames < 8) {
        audio_conv1d_compute_hwfma(job, NULL);
        return;
    }
    const size_t final_tap = (job->kernel_size - 1) * job->dilation;
    for (size_t oc = job->channel_begin; oc < job->channel_end; ++oc) {
        size_t frame = 0;
        for (; frame < job->output_frames;) {
            const int64_t first = (int64_t)frame - (int64_t)job->padding;
            /* Four independent vectors reuse each weight broadcast. Each lane
             * still visits input channels and taps in the scalar FMA order. */
            if (frame + 32 <= job->output_frames && first >= 0 &&
                first + 31 + (int64_t)final_tap < (int64_t)job->input_frames) {
                __m256 sums0 = _mm256_set1_ps(job->bias[oc]);
                __m256 sums1 = sums0;
                __m256 sums2 = sums0;
                __m256 sums3 = sums0;
                for (size_t ic = 0; ic < job->input_channels; ++ic) {
                    const float *row = job->weight +
                        (oc * job->input_channels + ic) * job->kernel_size;
                    const float *input = job->input + ic * job->input_stride +
                        (size_t)first;
                    for (size_t tap = 0; tap < job->kernel_size; ++tap) {
                        const float *values = input + tap * job->dilation;
                        const __m256 coefficient = _mm256_set1_ps(row[tap]);
                        sums0 = _mm256_fmadd_ps(_mm256_loadu_ps(values),
                                                 coefficient, sums0);
                        sums1 = _mm256_fmadd_ps(_mm256_loadu_ps(values + 8),
                                                 coefficient, sums1);
                        sums2 = _mm256_fmadd_ps(_mm256_loadu_ps(values + 16),
                                                 coefficient, sums2);
                        sums3 = _mm256_fmadd_ps(_mm256_loadu_ps(values + 24),
                                                 coefficient, sums3);
                    }
                }
                float *staged = job->staged + oc * job->output_frames + frame;
                _mm256_storeu_ps(staged, sums0);
                _mm256_storeu_ps(staged + 8, sums1);
                _mm256_storeu_ps(staged + 16, sums2);
                _mm256_storeu_ps(staged + 24, sums3);
                frame += 32;
                continue;
            }
            const int64_t last = first + 7 + (int64_t)final_tap;
            if (frame + 8 <= job->output_frames && first >= 0 &&
                last < (int64_t)job->input_frames) {
                __m256 sums = _mm256_set1_ps(job->bias[oc]);
                for (size_t ic = 0; ic < job->input_channels; ++ic) {
                    const float *row = job->weight +
                        (oc * job->input_channels + ic) * job->kernel_size;
                    const float *input = job->input + ic * job->input_stride +
                        (size_t)first;
                    for (size_t tap = 0; tap < job->kernel_size; ++tap) {
                        const __m256 values = _mm256_loadu_ps(input + tap * job->dilation);
                        sums = _mm256_fmadd_ps(values, _mm256_set1_ps(row[tap]), sums);
                    }
                }
                _mm256_storeu_ps(job->staged + oc * job->output_frames + frame, sums);
                frame += 8;
            } else {
                float sum = job->bias[oc];
                for (size_t ic = 0; ic < job->input_channels; ++ic) {
                    const float *row = job->weight +
                        (oc * job->input_channels + ic) * job->kernel_size;
                    for (size_t tap = 0; tap < job->kernel_size; ++tap) {
                        const int64_t source = (int64_t)frame +
                            (int64_t)tap * (int64_t)job->dilation -
                            (int64_t)job->padding;
                        if (source >= 0 && source < (int64_t)job->input_frames)
                            sum = fmaf(job->input[ic * job->input_stride +
                                (size_t)source], row[tap], sum);
                    }
                }
                job->staged[oc * job->output_frames + frame] = sum;
                ++frame;
            }
        }
    }
}
#endif
#undef CK_AUDIO_CONV1D_COMPUTE_BODY

int audio_conv1d_checked_hardware_fma_available(void) {
#if CK_AUDIO_CONV_X86_TARGET_FMA
    __builtin_cpu_init();
    return __builtin_cpu_supports("fma") != 0;
#else
    return 0;
#endif
}

int audio_conv1d_checked_avx2_fma_available(void) {
#if CK_AUDIO_CONV_X86_TARGET_FMA
    __builtin_cpu_init();
    return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
#else
    return 0;
#endif
}

int audio_conv1d_checked_workspace(size_t input_channels, size_t output_channels,
    size_t input_frames, size_t output_frames, size_t *scratch_elements) {
    size_t staged;
    if (!scratch_elements || !input_channels || !output_channels ||
        !input_frames || !output_frames ||
        input_channels > SIZE_MAX / input_frames ||
        output_channels > SIZE_MAX / output_frames)
        return CK_AUDIO_CONV_CHECKED_OVERFLOW;
    staged = output_channels * output_frames;
    if (staged > SIZE_MAX / sizeof(float))
        return CK_AUDIO_CONV_CHECKED_OVERFLOW;
    *scratch_elements = staged;
    return CK_AUDIO_CONV_CHECKED_OK;
}

int ck_audio_conv1d_checked_execute(
    const float *input, size_t input_elements, size_t input_stride,
    const float *weight, size_t weight_elements,
    const float *bias, size_t bias_elements,
    float *output, size_t output_elements, size_t output_stride,
    float *scratch, size_t scratch_elements,
    size_t input_channels, size_t output_channels, size_t input_frames,
    size_t kernel_size, size_t stride, size_t padding, size_t dilation,
    size_t output_frames,
    ck_audio_conv1d_compute_fn compute, void *compute_context) {
    size_t input_span, output_span, weight_count, scratch_need;
    if (!input_channels || !output_channels || !input_frames ||
        !kernel_size || !stride || !dilation || !output_frames ||
        input_channels > INT_MAX || output_channels > INT_MAX ||
        input_frames > INT_MAX || kernel_size > INT_MAX ||
        stride > INT_MAX || padding > INT_MAX || dilation > INT_MAX ||
        output_frames > INT_MAX)
        return CK_AUDIO_CONV_CHECKED_INVALID;
    const int64_t effective_kernel = 1 +
        ((int64_t)kernel_size - 1) * (int64_t)dilation;
    const int64_t padded = (int64_t)input_frames +
        2 * (int64_t)padding - effective_kernel;
    if (padded < 0 || padded / stride + 1 != output_frames)
        return CK_AUDIO_CONV_CHECKED_INVALID;
    if (!ck_checked_span((size_t)input_channels, (size_t)input_frames,
                         input_stride, &input_span) ||
        !ck_checked_span((size_t)output_channels, (size_t)output_frames,
                         output_stride, &output_span) ||
        audio_conv1d_checked_workspace((size_t)input_channels,
            (size_t)output_channels, (size_t)input_frames,
            (size_t)output_frames, &scratch_need))
        return CK_AUDIO_CONV_CHECKED_OVERFLOW;
    if ((size_t)output_channels > SIZE_MAX / (size_t)input_channels ||
        (size_t)output_channels * (size_t)input_channels > SIZE_MAX / (size_t)kernel_size)
        return CK_AUDIO_CONV_CHECKED_OVERFLOW;
    weight_count = (size_t)output_channels * (size_t)input_channels * (size_t)kernel_size;
    if (weight_count > SIZE_MAX / sizeof(float)) return CK_AUDIO_CONV_CHECKED_OVERFLOW;
    if (input_elements < input_span || weight_elements < weight_count ||
        bias_elements < (size_t)output_channels || output_elements < output_span ||
        scratch_elements < scratch_need)
        return CK_AUDIO_CONV_CHECKED_CAPACITY;
    if (!ck_checked_region(input, input_span) ||
        !ck_checked_region(weight, weight_count) ||
        !ck_checked_region(bias, (size_t)output_channels) ||
        !ck_checked_region(output, output_span) ||
        !ck_checked_region(scratch, scratch_need))
        return CK_AUDIO_CONV_CHECKED_INVALID;
    if (ck_checked_overlap(output, output_span, input, input_span) ||
        ck_checked_overlap(output, output_span, weight, weight_count) ||
        ck_checked_overlap(output, output_span, bias, (size_t)output_channels) ||
        ck_checked_overlap(output, output_span, scratch, scratch_need) ||
        ck_checked_overlap(scratch, scratch_need, input, input_span) ||
        ck_checked_overlap(scratch, scratch_need, weight, weight_count) ||
        ck_checked_overlap(scratch, scratch_need, bias, (size_t)output_channels))
        return CK_AUDIO_CONV_CHECKED_INVALID;
    if (!ck_checked_finite(weight, weight_count) ||
        !ck_checked_finite(bias, (size_t)output_channels))
        return CK_AUDIO_CONV_CHECKED_NONFINITE;
    for (size_t channel = 0; channel < input_channels; ++channel)
        if (!ck_checked_finite(input + (size_t)channel * input_stride,
                               (size_t)input_frames))
            return CK_AUDIO_CONV_CHECKED_NONFINITE;
    /* Validation and staged commit remain shared by both providers. */
    const ck_audio_conv1d_job job = {input, weight, bias, scratch,
        input_stride, input_channels, output_channels, input_frames,
        kernel_size, stride, padding, dilation, output_frames, 0,
        output_channels};
    compute(&job, compute_context);
    if (!ck_checked_finite(scratch, (size_t)output_channels * (size_t)output_frames))
        return CK_AUDIO_CONV_CHECKED_NONFINITE;
    for (size_t channel = 0; channel < output_channels; ++channel)
        memcpy(output + (size_t)channel * output_stride,
               scratch + (size_t)channel * (size_t)output_frames,
               (size_t)output_frames * sizeof(float));
    return CK_AUDIO_CONV_CHECKED_OK;
}

int audio_conv1d_checked_channel_major_f32(
    const float *input, size_t input_elements, size_t input_stride,
    const float *weight, size_t weight_elements,
    const float *bias, size_t bias_elements,
    float *output, size_t output_elements, size_t output_stride,
    float *scratch, size_t scratch_elements,
    size_t input_channels, size_t output_channels, size_t input_frames,
    size_t kernel_size, size_t stride, size_t padding, size_t output_frames) {
    return ck_audio_conv1d_checked_execute(input, input_elements, input_stride,
        weight, weight_elements, bias, bias_elements, output, output_elements,
        output_stride, scratch, scratch_elements, input_channels,
        output_channels, input_frames, kernel_size, stride, padding, 1,
        output_frames, audio_conv1d_compute_scalar, NULL);
}

int audio_conv1d_dilated_checked_channel_major_f32(
    const float *input, size_t input_elements, size_t input_stride,
    const float *weight, size_t weight_elements,
    const float *bias, size_t bias_elements,
    float *output, size_t output_elements, size_t output_stride,
    float *scratch, size_t scratch_elements,
    size_t input_channels, size_t output_channels, size_t input_frames,
    size_t kernel_size, size_t stride, size_t padding, size_t dilation,
    size_t output_frames) {
    return ck_audio_conv1d_checked_execute(input, input_elements, input_stride,
        weight, weight_elements, bias, bias_elements, output, output_elements,
        output_stride, scratch, scratch_elements, input_channels,
        output_channels, input_frames, kernel_size, stride, padding, dilation,
        output_frames, audio_conv1d_compute_scalar, NULL);
}

int audio_conv1d_checked_channel_major_f32_hwfma(
    const float *input, size_t input_elements, size_t input_stride,
    const float *weight, size_t weight_elements,
    const float *bias, size_t bias_elements,
    float *output, size_t output_elements, size_t output_stride,
    float *scratch, size_t scratch_elements,
    size_t input_channels, size_t output_channels, size_t input_frames,
    size_t kernel_size, size_t stride, size_t padding, size_t output_frames) {
#if CK_AUDIO_CONV_X86_TARGET_FMA
    if (!audio_conv1d_checked_hardware_fma_available())
        return CK_AUDIO_CONV_CHECKED_UNSUPPORTED;
    return ck_audio_conv1d_checked_execute(input, input_elements, input_stride,
        weight, weight_elements, bias, bias_elements, output, output_elements,
        output_stride, scratch, scratch_elements, input_channels,
        output_channels, input_frames, kernel_size, stride, padding, 1,
        output_frames, audio_conv1d_compute_hwfma, NULL);
#else
    (void)input; (void)input_elements; (void)input_stride;
    (void)weight; (void)weight_elements; (void)bias; (void)bias_elements;
    (void)output; (void)output_elements; (void)output_stride;
    (void)scratch; (void)scratch_elements; (void)input_channels;
    (void)output_channels; (void)input_frames; (void)kernel_size;
    (void)stride; (void)padding; (void)output_frames;
    return CK_AUDIO_CONV_CHECKED_UNSUPPORTED;
#endif
}

int audio_conv1d_dilated_checked_channel_major_f32_hwfma(
    const float *input, size_t input_elements, size_t input_stride,
    const float *weight, size_t weight_elements,
    const float *bias, size_t bias_elements,
    float *output, size_t output_elements, size_t output_stride,
    float *scratch, size_t scratch_elements,
    size_t input_channels, size_t output_channels, size_t input_frames,
    size_t kernel_size, size_t stride, size_t padding, size_t dilation,
    size_t output_frames) {
#if CK_AUDIO_CONV_X86_TARGET_FMA
    if (!audio_conv1d_checked_hardware_fma_available())
        return CK_AUDIO_CONV_CHECKED_UNSUPPORTED;
    return ck_audio_conv1d_checked_execute(input, input_elements, input_stride,
        weight, weight_elements, bias, bias_elements, output, output_elements,
        output_stride, scratch, scratch_elements, input_channels,
        output_channels, input_frames, kernel_size, stride, padding, dilation,
        output_frames, audio_conv1d_compute_hwfma, NULL);
#else
    (void)input; (void)input_elements; (void)input_stride;
    (void)weight; (void)weight_elements; (void)bias; (void)bias_elements;
    (void)output; (void)output_elements; (void)output_stride;
    (void)scratch; (void)scratch_elements; (void)input_channels;
    (void)output_channels; (void)input_frames; (void)kernel_size;
    (void)stride; (void)padding; (void)dilation; (void)output_frames;
    return CK_AUDIO_CONV_CHECKED_UNSUPPORTED;
#endif
}

int audio_conv1d_checked_channel_major_f32_avx2_fma(
    const float *input, size_t input_elements, size_t input_stride,
    const float *weight, size_t weight_elements,
    const float *bias, size_t bias_elements,
    float *output, size_t output_elements, size_t output_stride,
    float *scratch, size_t scratch_elements,
    size_t input_channels, size_t output_channels, size_t input_frames,
    size_t kernel_size, size_t stride, size_t padding, size_t output_frames) {
#if CK_AUDIO_CONV_X86_TARGET_FMA
    if (!audio_conv1d_checked_avx2_fma_available())
        return CK_AUDIO_CONV_CHECKED_UNSUPPORTED;
    return ck_audio_conv1d_checked_execute(input, input_elements, input_stride,
        weight, weight_elements, bias, bias_elements, output, output_elements,
        output_stride, scratch, scratch_elements, input_channels,
        output_channels, input_frames, kernel_size, stride, padding, 1,
        output_frames, audio_conv1d_compute_avx2, NULL);
#else
    (void)input; (void)input_elements; (void)input_stride;
    (void)weight; (void)weight_elements; (void)bias; (void)bias_elements;
    (void)output; (void)output_elements; (void)output_stride;
    (void)scratch; (void)scratch_elements; (void)input_channels;
    (void)output_channels; (void)input_frames; (void)kernel_size;
    (void)stride; (void)padding; (void)output_frames;
    return CK_AUDIO_CONV_CHECKED_UNSUPPORTED;
#endif
}

int audio_conv1d_dilated_checked_channel_major_f32_avx2_fma(
    const float *input, size_t input_elements, size_t input_stride,
    const float *weight, size_t weight_elements,
    const float *bias, size_t bias_elements,
    float *output, size_t output_elements, size_t output_stride,
    float *scratch, size_t scratch_elements,
    size_t input_channels, size_t output_channels, size_t input_frames,
    size_t kernel_size, size_t stride, size_t padding, size_t dilation,
    size_t output_frames) {
#if CK_AUDIO_CONV_X86_TARGET_FMA
    if (!audio_conv1d_checked_avx2_fma_available())
        return CK_AUDIO_CONV_CHECKED_UNSUPPORTED;
    return ck_audio_conv1d_checked_execute(input, input_elements, input_stride,
        weight, weight_elements, bias, bias_elements, output, output_elements,
        output_stride, scratch, scratch_elements, input_channels,
        output_channels, input_frames, kernel_size, stride, padding, dilation,
        output_frames, audio_conv1d_compute_avx2, NULL);
#else
    (void)input; (void)input_elements; (void)input_stride;
    (void)weight; (void)weight_elements; (void)bias; (void)bias_elements;
    (void)output; (void)output_elements; (void)output_stride;
    (void)scratch; (void)scratch_elements; (void)input_channels;
    (void)output_channels; (void)input_frames; (void)kernel_size;
    (void)stride; (void)padding; (void)dilation; (void)output_frames;
    return CK_AUDIO_CONV_CHECKED_UNSUPPORTED;
#endif
}

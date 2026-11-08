#include "ckernel_linear_checked.h"

#include <float.h>
#include <math.h>
#include <stdint.h>

static int span_fits(size_t rows, size_t stride, size_t width, size_t elements)
{
    return rows && width && stride >= width &&
           rows - 1 <= (SIZE_MAX - width) / stride &&
           (rows - 1) * stride + width <= elements;
}

static double dot_bias(const float *input, const float *weight,
                       size_t width, float bias)
{
    double sum = (double)bias;
    for (size_t c = 0; c < width; ++c) {
        volatile double product = (double)input[c] * (double)weight[c];
        sum += product;
    }
    return sum;
}


#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
/* Independent rows share weight loads; tail rows vectorize output channels.
 * Each lane preserves ordered FP64
 * multiplication and addition. No horizontal reduction or FMA is introduced. */
__attribute__((target("avx2,no-fma"), noinline))
static void compute_independent_lanes(const float *input, size_t input_stride,
                              const float *weight, size_t weight_stride,
                              const float *bias, float *output, size_t output_stride,
                              size_t rows, size_t width, size_t channels)
{
    size_t r = 0;
    for (; r + 4 <= rows; r += 4) {
        for (size_t o = 0; o < channels; ++o) {
            __m256d sum = _mm256_set1_pd(bias ? (double)bias[o] : 0.0);
            size_t c = 0;
            for (; c + 4 <= width; c += 4) {
                __m128 a = _mm_loadu_ps(input + r * input_stride + c);
                __m128 b = _mm_loadu_ps(input + (r + 1) * input_stride + c);
                __m128 d = _mm_loadu_ps(input + (r + 2) * input_stride + c);
                __m128 e = _mm_loadu_ps(input + (r + 3) * input_stride + c);
                _MM_TRANSPOSE4_PS(a, b, d, e);
                sum = _mm256_add_pd(sum, _mm256_mul_pd(_mm256_cvtps_pd(a),
                    _mm256_set1_pd((double)weight[o * weight_stride + c])));
                sum = _mm256_add_pd(sum, _mm256_mul_pd(_mm256_cvtps_pd(b),
                    _mm256_set1_pd((double)weight[o * weight_stride + c + 1])));
                sum = _mm256_add_pd(sum, _mm256_mul_pd(_mm256_cvtps_pd(d),
                    _mm256_set1_pd((double)weight[o * weight_stride + c + 2])));
                sum = _mm256_add_pd(sum, _mm256_mul_pd(_mm256_cvtps_pd(e),
                    _mm256_set1_pd((double)weight[o * weight_stride + c + 3])));
            }
            for (; c < width; ++c) {
                __m256d values = _mm256_set_pd(input[(r+3)*input_stride+c],
                    input[(r+2)*input_stride+c], input[(r+1)*input_stride+c],
                    input[r*input_stride+c]);
                __m256d w = _mm256_set1_pd((double)weight[o*weight_stride+c]);
                sum = _mm256_add_pd(sum, _mm256_mul_pd(values, w));
            }
            double lanes[4];
            _mm256_storeu_pd(lanes, sum);
            for (size_t lane = 0; lane < 4; ++lane)
                output[(r+lane)*output_stride+o] = (float)lanes[lane];
        }
    }
    for (; r < rows; ++r) {
        size_t o = 0;
        for (; o + 4 <= channels; o += 4) {
            __m256d sum = bias ? _mm256_cvtps_pd(_mm_loadu_ps(bias + o)) :
                                _mm256_setzero_pd();
            size_t c = 0;
            for (; c + 4 <= width; c += 4) {
                __m128 a = _mm_loadu_ps(weight + o * weight_stride + c);
                __m128 b = _mm_loadu_ps(weight + (o + 1) * weight_stride + c);
                __m128 d = _mm_loadu_ps(weight + (o + 2) * weight_stride + c);
                __m128 e = _mm_loadu_ps(weight + (o + 3) * weight_stride + c);
                _MM_TRANSPOSE4_PS(a, b, d, e);
                sum = _mm256_add_pd(sum, _mm256_mul_pd(_mm256_cvtps_pd(a),
                    _mm256_set1_pd((double)input[r * input_stride + c])));
                sum = _mm256_add_pd(sum, _mm256_mul_pd(_mm256_cvtps_pd(b),
                    _mm256_set1_pd((double)input[r * input_stride + c + 1])));
                sum = _mm256_add_pd(sum, _mm256_mul_pd(_mm256_cvtps_pd(d),
                    _mm256_set1_pd((double)input[r * input_stride + c + 2])));
                sum = _mm256_add_pd(sum, _mm256_mul_pd(_mm256_cvtps_pd(e),
                    _mm256_set1_pd((double)input[r * input_stride + c + 3])));
            }
            for (; c < width; ++c) {
                const __m256d w = _mm256_set_pd(weight[(o+3)*weight_stride+c],
                    weight[(o+2)*weight_stride+c], weight[(o+1)*weight_stride+c],
                    weight[o*weight_stride+c]);
                const __m256d value = _mm256_set1_pd((double)input[r*input_stride+c]);
                sum = _mm256_add_pd(sum, _mm256_mul_pd(value, w));
            }
            _mm_storeu_ps(output+r*output_stride+o, _mm256_cvtpd_ps(sum));
        }
        for (; o < channels; ++o)
            output[r*output_stride+o] = (float)dot_bias(input+r*input_stride,
                weight+o*weight_stride, width, bias ? bias[o] : 0.0f);
    }
}

/* Validate four independent weight rows with the original per-row L1 order.
 * No output is published here; a late nonfinite value still rejects atomically. */
__attribute__((target("avx2,no-fma"), noinline))
static int validate_weight_rows(const float *weight, size_t stride,
                                const float *bias, size_t width, size_t channels,
                                double *max_l1, double *max_bias)
{
    size_t o = 0;
    const __m256d sign = _mm256_set1_pd(-0.0);
    const __m256d limit = _mm256_set1_pd((double)FLT_MAX);
    for (; o + 4 <= channels; o += 4) {
        __m256d sums = _mm256_setzero_pd();
        size_t c = 0;
        for (; c + 4 <= width; c += 4) {
            const __m128 sign32 = _mm_set1_ps(-0.0f);
            const __m128 limit32 = _mm_set1_ps(FLT_MAX);
            __m128 a = _mm_andnot_ps(sign32, _mm_loadu_ps(weight + o * stride + c));
            __m128 b = _mm_andnot_ps(sign32, _mm_loadu_ps(weight + (o + 1) * stride + c));
            __m128 d = _mm_andnot_ps(sign32, _mm_loadu_ps(weight + (o + 2) * stride + c));
            __m128 e = _mm_andnot_ps(sign32, _mm_loadu_ps(weight + (o + 3) * stride + c));
            const __m128 finite = _mm_and_ps(_mm_and_ps(_mm_cmple_ps(a, limit32),
                _mm_cmple_ps(b, limit32)), _mm_and_ps(_mm_cmple_ps(d, limit32),
                _mm_cmple_ps(e, limit32)));
            if (_mm_movemask_ps(finite) != 15) return -1;
            _MM_TRANSPOSE4_PS(a, b, d, e);
            sums = _mm256_add_pd(sums, _mm256_cvtps_pd(a));
            sums = _mm256_add_pd(sums, _mm256_cvtps_pd(b));
            sums = _mm256_add_pd(sums, _mm256_cvtps_pd(d));
            sums = _mm256_add_pd(sums, _mm256_cvtps_pd(e));
        }
        for (; c < width; ++c) {
            __m256d v = _mm256_set_pd(weight[(o+3)*stride+c], weight[(o+2)*stride+c],
                                     weight[(o+1)*stride+c], weight[o*stride+c]);
            v = _mm256_andnot_pd(sign, v);
            if (_mm256_movemask_pd(_mm256_cmp_pd(v, limit, _CMP_LE_OQ)) != 15)
                return -1;
            sums = _mm256_add_pd(sums, v);
        }
        double lanes[4];
        _mm256_storeu_pd(lanes, sums);
        for (size_t lane = 0; lane < 4; ++lane) {
            if (lanes[lane] > *max_l1) *max_l1 = lanes[lane];
            if (bias) {
                if (!isfinite(bias[o+lane])) return -1;
                const double b = fabs((double)bias[o+lane]);
                if (b > *max_bias) *max_bias = b;
            }
        }
    }
    for (; o < channels; ++o) {
        double sum = 0.0;
        for (size_t c = 0; c < width; ++c) {
            const float v = weight[o*stride+c];
            if (!isfinite(v)) return -1;
            sum += fabs((double)v);
        }
        if (sum > *max_l1) *max_l1 = sum;
        if (bias) {
            if (!isfinite(bias[o])) return -1;
            const double b = fabs((double)bias[o]);
            if (b > *max_bias) *max_bias = b;
        }
    }
    return 0;
}

static int disjoint(const float *a, size_t na, const float *b, size_t nb)
{
    if (!b || !nb) return 1;
    uintptr_t x = (uintptr_t)a, y = (uintptr_t)b;
    if (na > (UINTPTR_MAX-x)/sizeof(float) || nb > (UINTPTR_MAX-y)/sizeof(float))
        return 0;
    return x + na*sizeof(float) <= y || y + nb*sizeof(float) <= x;
}
#endif

int linear_rows_checked_f32(
    const float *input, size_t input_elements, size_t input_stride,
    const float *weight, size_t weight_elements, size_t weight_stride,
    const float *bias, size_t bias_elements,
    float *output, size_t output_elements, size_t output_stride,
    size_t rows, size_t input_channels, size_t output_channels)
{
    if (!input || !weight || !output ||
        !span_fits(rows, input_stride, input_channels, input_elements) ||
        !span_fits(output_channels, weight_stride, input_channels, weight_elements) ||
        !span_fits(rows, output_stride, output_channels, output_elements) ||
        (bias ? bias_elements < output_channels : bias_elements != 0))
        return -1;

    /* Validate every accessed value before the first write. A conservative
     * absolute bound permits one compute pass for ordinary model values.
     * Values near the FP32 limit retain the exact two-pass staging path. */
    double max_input = 0.0;
    double max_weight_l1 = 0.0;
    double max_bias = 0.0;
    for (size_t r = 0; r < rows; ++r)
        for (size_t c = 0; c < input_channels; ++c)
        {
            const float value = input[r * input_stride + c];
            if (!isfinite(value)) return -1;
            const double absolute = fabs((double)value);
            if (absolute > max_input) max_input = absolute;
        }
#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
    __builtin_cpu_init();
    if (output_channels >= 4 && __builtin_cpu_supports("avx2")) {
        if (validate_weight_rows(weight, weight_stride, bias, input_channels,
                                 output_channels, &max_weight_l1, &max_bias))
            return -1;
    } else
#endif
    for (size_t o = 0; o < output_channels; ++o) {
        if (bias && !isfinite(bias[o])) return -1;
        if (bias && fabs((double)bias[o]) > max_bias)
            max_bias = fabs((double)bias[o]);
        double weight_l1 = 0.0;
        for (size_t c = 0; c < input_channels; ++c)
        {
            const float value = weight[o * weight_stride + c];
            if (!isfinite(value)) return -1;
            weight_l1 += fabs((double)value);
        }
        if (weight_l1 > max_weight_l1) max_weight_l1 = weight_l1;
    }
    /* At most UINT32_MAX positive terms keep FP64 summation error far below
     * the 4x guard, including error in the L1 bound. Every result therefore
     * fits FP32 and publishing cannot fail after the first output write. */
    if (input_channels <= (SIZE_MAX / 4) && input_channels <= UINT32_MAX &&
        max_input * max_weight_l1 + max_bias <= (double)FLT_MAX / 4.0) {
#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
        __builtin_cpu_init();
        if ((rows >= 4 || output_channels >= 4) && __builtin_cpu_supports("avx2") &&
            disjoint(output, output_elements, input, input_elements) &&
            disjoint(output, output_elements, weight, weight_elements) &&
            disjoint(output, output_elements, bias, bias_elements)) {
            compute_independent_lanes(input, input_stride, weight, weight_stride, bias,
                              output, output_stride, rows, input_channels,
                              output_channels);
            return 0;
        }
#endif
        for (size_t r = 0; r < rows; ++r)
            for (size_t o = 0; o < output_channels; ++o)
                output[r * output_stride + o] = (float)dot_bias(
                    input + r * input_stride, weight + o * weight_stride,
                    input_channels, bias ? bias[o] : 0.0f);
        return 0;
    }
    for (size_t r = 0; r < rows; ++r)
        for (size_t o = 0; o < output_channels; ++o) {
            const double result = dot_bias(input + r * input_stride,
                                           weight + o * weight_stride,
                                           input_channels, bias ? bias[o] : 0.0f);
            if (!isfinite(result) || fabs(result) > FLT_MAX) return -1;
        }
    for (size_t r = 0; r < rows; ++r)
        for (size_t o = 0; o < output_channels; ++o)
            output[r * output_stride + o] = (float)dot_bias(
                input + r * input_stride, weight + o * weight_stride,
                input_channels, bias ? bias[o] : 0.0f);
    return 0;
}

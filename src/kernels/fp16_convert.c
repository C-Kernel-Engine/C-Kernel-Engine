/**
 * @file fp16_convert.c
 * @brief FP32 <-> FP16 SIMD conversion utilities
 *
 * CK-ENGINE KERNEL RULES:
 * =======================
 * 1. NO malloc/free - memory via bump allocator, pointers passed in
 * 2. NO OpenMP - parallelization at orchestrator/codegen layer
 * 3. NO memcpy for layout - use strided access, not copies
 * 4. API must define: inputs, outputs, workspace, and memory layouts
 * 5. Pure computation - deterministic, no side effects
 *
 * After changes: make test && make llamacpp-parity-full
 *
 * These conversion functions use F16C hardware instructions (available on
 * Intel Ivy Bridge and later, AMD Piledriver and later) for fast FP16/FP32
 * conversion. FP16 (IEEE 754 half-precision) provides 2x memory savings
 * with ~0.1% precision loss for KV cache storage.
 *
 * MEGA-FUSION BENEFIT:
 * ====================
 * FP16 KV cache doubles the context that fits in L3 cache:
 *   - FP32 KV: ~6K tokens in 6MB L3
 *   - FP16 KV: ~12K tokens in 6MB L3
 * This extends mega-fusion's "hot zone" for longer sequences.
 */

#include <stdint.h>
#include <stddef.h>
#include <math.h>

#if defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__)
#include <immintrin.h>
#endif

/* ============================================================================
 * Scalar FP16 <-> FP32 Conversion
 * ============================================================================
 * These use F16C instructions when available, with software fallback.
 */

#if defined(__F16C__) || (defined(__AVX__) && !defined(__clang__))
/* Hardware F16C support */

/**
 * @brief Convert single FP32 to FP16 (hardware)
 * @param f FP32 value
 * @return FP16 value as uint16_t
 */
static inline uint16_t ck_fp32_to_fp16_scalar(float f) {
    return _cvtss_sh(f, _MM_FROUND_TO_NEAREST_INT);
}

/**
 * @brief Convert single FP16 to FP32 (hardware)
 * @param h FP16 value as uint16_t
 * @return FP32 value
 */
static inline float ck_fp16_to_fp32_scalar(uint16_t h) {
    return _cvtsh_ss(h);
}

#else
/* Software fallback for systems without F16C */

static inline uint16_t ck_fp32_to_fp16_scalar(float f) {
    union { float f; uint32_t u; } u = { f };
    uint32_t x = u.u;

    /* Extract sign, exponent, mantissa */
    uint32_t sign = (x >> 16) & 0x8000;
    int exp = ((x >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = (x >> 13) & 0x3FF;

    if (exp <= 0) {
        /* Underflow to zero or denormal */
        if (exp < -10) return (uint16_t)sign;
        mant = (mant | 0x400) >> (1 - exp);
        return (uint16_t)(sign | mant);
    } else if (exp >= 31) {
        /* Overflow to infinity or NaN */
        if (exp == 128 && (x & 0x7FFFFF)) {
            return (uint16_t)(sign | 0x7E00 | mant);  /* NaN */
        }
        return (uint16_t)(sign | 0x7C00);  /* Infinity */
    }

    return (uint16_t)(sign | ((uint32_t)exp << 10) | mant);
}

static inline float ck_fp16_to_fp32_scalar(uint16_t h) {
    uint32_t sign = ((uint32_t)h & 0x8000) << 16;
    int exp = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;

    if (exp == 0) {
        if (mant == 0) {
            /* Zero */
            union { uint32_t u; float f; } u = { sign };
            return u.f;
        }
        /* Denormalized number */
        while (!(mant & 0x400)) {
            mant <<= 1;
            exp--;
        }
        exp++;
        mant &= 0x3FF;
    } else if (exp == 31) {
        /* Infinity or NaN */
        union { uint32_t u; float f; } u = { sign | 0x7F800000 | (mant << 13) };
        return u.f;
    }

    union { uint32_t u; float f; } u = { sign | ((uint32_t)(exp + 112) << 23) | (mant << 13) };
    return u.f;
}
#endif

/* ============================================================================
 * Vectorized FP32 -> FP16 Conversion
 * ============================================================================ */

#if defined(__AVX512F__)
/**
 * @brief Convert FP32 array to FP16 using AVX-512 (16 floats at a time)
 * @param src Source FP32 array
 * @param dst Destination FP16 array
 * @param n Number of elements
 */
void ck_fp32_to_fp16_avx512(const float *src, uint16_t *dst, int n) {
    if (!src || !dst || n <= 0) return;

    int i = 0;
    for (; i + 15 < n; i += 16) {
        __m512 x = _mm512_loadu_ps(src + i);
        __m256i y = _mm512_cvtps_ph(x, _MM_FROUND_TO_NEAREST_INT);
        _mm256_storeu_si256((__m256i*)(dst + i), y);
    }

    /* Handle remaining elements */
    for (; i < n; i++) {
        dst[i] = ck_fp32_to_fp16_scalar(src[i]);
    }
}

/**
 * @brief Convert FP16 array to FP32 using AVX-512 (16 floats at a time)
 * @param src Source FP16 array
 * @param dst Destination FP32 array
 * @param n Number of elements
 */
void ck_fp16_to_fp32_avx512(const uint16_t *src, float *dst, int n) {
    if (!src || !dst || n <= 0) return;

    int i = 0;
    for (; i + 15 < n; i += 16) {
        __m256i x = _mm256_loadu_si256((const __m256i*)(src + i));
        __m512 y = _mm512_cvtph_ps(x);
        _mm512_storeu_ps(dst + i, y);
    }

    /* Handle remaining elements */
    for (; i < n; i++) {
        dst[i] = ck_fp16_to_fp32_scalar(src[i]);
    }
}
#endif /* __AVX512F__ */

#if defined(__AVX__)
/**
 * @brief Convert FP32 array to FP16 using AVX + F16C (8 floats at a time)
 * @param src Source FP32 array
 * @param dst Destination FP16 array
 * @param n Number of elements
 */
void ck_fp32_to_fp16_avx(const float *src, uint16_t *dst, int n) {
    if (!src || !dst || n <= 0) return;

    int i = 0;
#if defined(__F16C__)
    for (; i + 7 < n; i += 8) {
        __m256 x = _mm256_loadu_ps(src + i);
        __m128i y = _mm256_cvtps_ph(x, _MM_FROUND_TO_NEAREST_INT);
        _mm_storeu_si128((__m128i*)(dst + i), y);
    }
#endif

    /* Handle remaining elements */
    for (; i < n; i++) {
        dst[i] = ck_fp32_to_fp16_scalar(src[i]);
    }
}

/**
 * @brief Convert FP16 array to FP32 using AVX + F16C (8 floats at a time)
 * @param src Source FP16 array
 * @param dst Destination FP32 array
 * @param n Number of elements
 */
void ck_fp16_to_fp32_avx(const uint16_t *src, float *dst, int n) {
    if (!src || !dst || n <= 0) return;

    int i = 0;
#if defined(__F16C__)
    for (; i + 7 < n; i += 8) {
        __m128i x = _mm_loadu_si128((const __m128i*)(src + i));
        __m256 y = _mm256_cvtph_ps(x);
        _mm256_storeu_ps(dst + i, y);
    }
#endif

    /* Handle remaining elements */
    for (; i < n; i++) {
        dst[i] = ck_fp16_to_fp32_scalar(src[i]);
    }
}
#endif /* __AVX__ */

/* ============================================================================
 * Generic Dispatch Functions
 * ============================================================================ */

/**
 * @brief Convert FP32 row to FP16 (auto-select best implementation)
 * @param src Source FP32 array
 * @param dst Destination FP16 array (caller-allocated)
 * @param n Number of elements
 */
void ck_fp32_to_fp16_row(const float *src, uint16_t *dst, int n) {
    if (!src || !dst || n <= 0) return;

#if defined(__AVX512F__)
    ck_fp32_to_fp16_avx512(src, dst, n);
#elif defined(__AVX__)
    ck_fp32_to_fp16_avx(src, dst, n);
#else
    for (int i = 0; i < n; i++) {
        dst[i] = ck_fp32_to_fp16_scalar(src[i]);
    }
#endif
}

/**
 * @brief Convert FP16 row to FP32 (auto-select best implementation)
 * @param src Source FP16 array
 * @param dst Destination FP32 array (caller-allocated)
 * @param n Number of elements
 */
void ck_fp16_to_fp32_row(const uint16_t *src, float *dst, int n) {
    if (!src || !dst || n <= 0) return;

#if defined(__AVX512F__)
    ck_fp16_to_fp32_avx512(src, dst, n);
#elif defined(__AVX__)
    ck_fp16_to_fp32_avx(src, dst, n);
#else
    for (int i = 0; i < n; i++) {
        dst[i] = ck_fp16_to_fp32_scalar(src[i]);
    }
#endif
}

/* ============================================================================
 * 2D Conversion Functions (for matrices)
 * ============================================================================ */

/**
 * @brief Convert 2D FP32 matrix to FP16 with strided access
 * @param src Source FP32 matrix [rows, src_stride]
 * @param dst Destination FP16 matrix [rows, dst_stride]
 * @param rows Number of rows
 * @param cols Number of columns (actual data per row)
 * @param src_stride Source stride (elements per row)
 * @param dst_stride Destination stride (elements per row)
 */
void ck_fp32_to_fp16_2d(const float *src, uint16_t *dst,
                         int rows, int cols,
                         int src_stride, int dst_stride) {
    if (!src || !dst || rows <= 0 || cols <= 0) return;

    for (int r = 0; r < rows; r++) {
        ck_fp32_to_fp16_row(src + (size_t)r * src_stride,
                            dst + (size_t)r * dst_stride,
                            cols);
    }
}

/**
 * @brief Convert 2D FP16 matrix to FP32 with strided access
 * @param src Source FP16 matrix [rows, src_stride]
 * @param dst Destination FP32 matrix [rows, dst_stride]
 * @param rows Number of rows
 * @param cols Number of columns (actual data per row)
 * @param src_stride Source stride (elements per row)
 * @param dst_stride Destination stride (elements per row)
 */
void ck_fp16_to_fp32_2d(const uint16_t *src, float *dst,
                         int rows, int cols,
                         int src_stride, int dst_stride) {
    if (!src || !dst || rows <= 0 || cols <= 0) return;

    for (int r = 0; r < rows; r++) {
        ck_fp16_to_fp32_row(src + (size_t)r * src_stride,
                            dst + (size_t)r * dst_stride,
                            cols);
    }
}

/* ============================================================================
 * In-place Conversion with Scratch Buffer
 * ============================================================================ */

/**
 * @brief Convert FP32 to FP16 in-place using scratch buffer
 *
 * Useful when you want to downcast in place but need FP32 for computation.
 * Writes FP16 to the lower half of scratch, then copies back.
 *
 * @param data FP32 array to convert (will contain FP16 in lower bits)
 * @param scratch Temporary buffer, must be >= n * sizeof(uint16_t)
 * @param n Number of elements
 * @note After this call, data should be treated as uint16_t*
 */
void ck_fp32_to_fp16_inplace(float *data, void *scratch, int n) {
    if (!data || !scratch || n <= 0) return;

    uint16_t *tmp = (uint16_t*)scratch;
    ck_fp32_to_fp16_row(data, tmp, n);

    /* Copy back (FP16 is half the size, so this is safe) */
    uint16_t *dst = (uint16_t*)data;
    for (int i = 0; i < n; i++) {
        dst[i] = tmp[i];
    }
}

/* ============================================================================
 * Mixed Precision Operations (compute in FP32, store in FP16)
 * ============================================================================ */

/**
 * @brief FMA in FP32, store result as FP16: dst = a * b + c
 * @param a First FP32 operand array
 * @param b Second FP32 operand array
 * @param c Third FP32 operand array
 * @param dst Destination FP16 array
 * @param n Number of elements
 */
void ck_fma_f32_to_f16(const float *a, const float *b, const float *c,
                        uint16_t *dst, int n) {
    if (!a || !b || !c || !dst || n <= 0) return;

#if defined(__AVX512F__)
    int i = 0;
    for (; i + 15 < n; i += 16) {
        __m512 va = _mm512_loadu_ps(a + i);
        __m512 vb = _mm512_loadu_ps(b + i);
        __m512 vc = _mm512_loadu_ps(c + i);
        __m512 vr = _mm512_fmadd_ps(va, vb, vc);
        __m256i vh = _mm512_cvtps_ph(vr, _MM_FROUND_TO_NEAREST_INT);
        _mm256_storeu_si256((__m256i*)(dst + i), vh);
    }
    for (; i < n; i++) {
        dst[i] = ck_fp32_to_fp16_scalar(a[i] * b[i] + c[i]);
    }
#elif defined(__AVX__) && defined(__F16C__)
    int i = 0;
    for (; i + 7 < n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 vc = _mm256_loadu_ps(c + i);
#if defined(__FMA__)
        __m256 vr = _mm256_fmadd_ps(va, vb, vc);
#else
        __m256 vr = _mm256_add_ps(_mm256_mul_ps(va, vb), vc);
#endif
        __m128i vh = _mm256_cvtps_ph(vr, _MM_FROUND_TO_NEAREST_INT);
        _mm_storeu_si128((__m128i*)(dst + i), vh);
    }
    for (; i < n; i++) {
        dst[i] = ck_fp32_to_fp16_scalar(a[i] * b[i] + c[i]);
    }
#else
    for (int i = 0; i < n; i++) {
        dst[i] = ck_fp32_to_fp16_scalar(a[i] * b[i] + c[i]);
    }
#endif
}

/**
 * @brief Scale FP32 array and store as FP16: dst = scale * src
 * @param src Source FP32 array
 * @param scale Scalar multiplier
 * @param dst Destination FP16 array
 * @param n Number of elements
 */
void ck_scale_f32_to_f16(const float *src, float scale, uint16_t *dst, int n) {
    if (!src || !dst || n <= 0) return;

#if defined(__AVX512F__)
    __m512 vs = _mm512_set1_ps(scale);
    int i = 0;
    for (; i + 15 < n; i += 16) {
        __m512 vx = _mm512_loadu_ps(src + i);
        __m512 vr = _mm512_mul_ps(vx, vs);
        __m256i vh = _mm512_cvtps_ph(vr, _MM_FROUND_TO_NEAREST_INT);
        _mm256_storeu_si256((__m256i*)(dst + i), vh);
    }
    for (; i < n; i++) {
        dst[i] = ck_fp32_to_fp16_scalar(src[i] * scale);
    }
#elif defined(__AVX__) && defined(__F16C__)
    __m256 vs = _mm256_set1_ps(scale);
    int i = 0;
    for (; i + 7 < n; i += 8) {
        __m256 vx = _mm256_loadu_ps(src + i);
        __m256 vr = _mm256_mul_ps(vx, vs);
        __m128i vh = _mm256_cvtps_ph(vr, _MM_FROUND_TO_NEAREST_INT);
        _mm_storeu_si128((__m128i*)(dst + i), vh);
    }
    for (; i < n; i++) {
        dst[i] = ck_fp32_to_fp16_scalar(src[i] * scale);
    }
#else
    for (int i = 0; i < n; i++) {
        dst[i] = ck_fp32_to_fp16_scalar(src[i] * scale);
    }
#endif
}

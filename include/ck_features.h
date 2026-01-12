/**
 * @file ck_features.h
 * @brief CPU feature detection and dispatch macros
 *
 * Defines standardized macros for SIMD instruction set detection
 * and kernel dispatch. Use these instead of CPU model checks.
 *
 * Feature Priority (best available):
 *   AMX (512-bit tile ops, Intel Sapphire Rapids+)
 *   AVX-512 (512-bit vector, Intel Skylake-X+)
 *   AVX2 (256-bit with FMA, Intel Haswell+)
 *   AVX (256-bit, Intel Sandy Bridge+)
 *   NEON/SVE2 (ARM)
 *   DSA (PowerPC)
 *   Reference (fallback)
 */

#ifndef CK_FEATURES_H
#define CK_FEATURES_H

#include <stdint.h>

/*============================================================================
 * Compiler Feature Detection
 * These are set by the compiler based on -march flags
 *============================================================================*/

/* Intel/x86 */
#if defined(__AMX__)
    #define CK_HAS_AMX 1
#endif

#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512DQ__)
    #define CK_HAS_AVX512 1
#endif

#if defined(__AVX2__) && defined(__FMA__)
    #define CK_HAS_AVX2_FMA 1
#endif

#if defined(__AVX__)
    #define CK_HAS_AVX 1
#endif

#if defined(__VNNI__)
    #define CK_HAS_VNNI 1
#endif

#if defined(__AVX512VNNI__)
    #define CK_HAS_AVX512VNNI 1
#endif

/* ARM */
#if defined(__aarch64__)
    #if defined(__ARM_FEATURE_SVE2)
        #define CK_HAS_SVE2 1
    #endif
    #if defined(__ARM_FEATURE_NEON)
        #define CK_HAS_NEON 1
    #endif
#endif

/* PowerPC */
#if defined(__ALTIVEC__)
    #define CK_HAS_ALTIVEC 1
#endif

#if defined(__VSX__)
    #define CK_HAS_VSX 1
#endif

/* RISC-V */
#if defined(__riscv_vector)
    #define CK_HAS_RVV 1
#endif

/*============================================================================
 * Runtime Feature Detection (CPUID/MSR)
 * For compiled binaries that need runtime dispatch
 *============================================================================*/

#if defined(__x86_64__) || defined(__i386__)
    #include <cpuid.h>

    static inline uint32_t ck_cpuid_max_leaf(void) {
        uint32_t eax, ecx = 0, ebx = 0, edx = 0;
        __cpuid(0, eax, ebx, ecx, edx);
        return eax;
    }

    static inline void ck_cpuid_leaf1(uint32_t *eax, uint32_t *ebx,
                                       uint32_t *ecx, uint32_t *edx) {
        __cpuid(1, *eax, *ebx, *ecx, *edx);
    }

    static inline void ck_cpuid_leaf7(uint32_t ecx_val, uint32_t *eax,
                                       uint32_t *ebx, uint32_t *ecx,
                                       uint32_t *edx) {
        int info[4];
        __cpuidex(info, 7, ecx_val);
        *eax = info[0];
        *ebx = info[1];
        *ecx = info[2];
        *edx = info[3];
    }

    /* Check if OS supports XSAVE/XRSTORE (needed for AVX-512/AMX state) */
    static inline int ck_os_has_xtile(void) {
        uint32_t eax, ebx, ecx, edx;
        ck_cpuid_leaf1(&eax, &ebx, &ecx, &edx);
        /* ECX bit 26 = XSAVE, ECX bit 27 = OSXSAVE */
        return (ecx & (1 << 27)) && (ecx & (1 << 26));
    }
#endif

/*============================================================================
 * Feature Flag Macros
 * Use these in dispatch functions instead of CPU model checks
 *============================================================================*/

/* Best available vector width */
#if defined(CK_HAS_AMX)
    #define CK_VECTOR_WIDTH 512
    #define CK_HAS_BEST_VECTOR 1
#elif defined(CK_HAS_AVX512)
    #define CK_VECTOR_WIDTH 512
    #define CK_HAS_BEST_VECTOR 1
#elif defined(CK_HAS_AVX2_FMA)
    #define CK_VECTOR_WIDTH 256
    #define CK_HAS_BEST_VECTOR 1
#elif defined(CK_HAS_AVX)
    #define CK_VECTOR_WIDTH 256
    #define CK_HAS_BEST_VECTOR 1
#elif defined(CK_HAS_NEON)
    #define CK_VECTOR_WIDTH 128
    #define CK_HAS_BEST_VECTOR 1
#else
    #define CK_VECTOR_WIDTH 32  /* Scalar fallback */
    #define CK_HAS_BEST_VECTOR 0
#endif

/* AI acceleration features */
#if defined(CK_HAS_AMX)
    #define CK_HAS_AI_ACCEL 1
#elif defined(CK_HAS_AVX512VNNI)
    #define CK_HAS_AI_ACCEL 1
#elif defined(CK_HAS_VNNI)
    #define CK_HAS_AI_ACCEL 1
#else
    #define CK_HAS_AI_ACCEL 0
#endif

/*============================================================================
 * Kernel Dispatch Macros
 * Use these for clean dispatch in kernel functions
 *============================================================================*/

/**
 * @brief Dispatch to best available GEMM kernel
 *
 * Usage:
 *   CK_GEMM_DISPATCH(y, W, x, M, K);
 *   expands to appropriate kernel call
 */
#if defined(CK_HAS_AMX)
    #define CK_GEMM_DISPATCH(...) gemm_amx(__VA_ARGS__)
#elif defined(CK_HAS_AVX512)
    #define CK_GEMM_DISPATCH(...) gemm_avx512(__VA_ARGS__)
#elif defined(CK_HAS_AVX2_FMA)
    #define CK_GEMM_DISPATCH(...) gemm_avx2(__VA_ARGS__)
#elif defined(CK_HAS_AVX)
    #define CK_GEMM_DISPATCH(...) gemm_avx(__VA_ARGS__)
#else
    #define CK_GEMM_DISPATCH(...) gemm_ref(__VA_ARGS__)
#endif

/**
 * @brief Dispatch to best available GEMV kernel
 */
#if defined(CK_HAS_AMX)
    #define CK_GEMV_DISPATCH(...) gemv_amx(__VA_ARGS__)
#elif defined(CK_HAS_AVX512)
    #define CK_GEMV_DISPATCH(...) gemv_avx512(__VA_ARGS__)
#elif defined(CK_HAS_AVX2_FMA)
    #define CK_GEMV_DISPATCH(...) gemv_avx2(__VA_ARGS__)
#elif defined(CK_HAS_AVX)
    #define CK_GEMV_DISPATCH(...) gemv_avx(__VA_ARGS__)
#else
    #define CK_GEMV_DISPATCH(...) gemv_ref(__VA_ARGS__)
#endif

/**
 * @brief Dispatch to best available quantized GEMV kernel
 * For INT8/INT4 quantization with VNNI/AMX acceleration
 */
#if defined(CK_HAS_AMX)
    #define CK_QGEMM_DISPATCH(...) qgemm_amx(__VA_ARGS__)
#elif defined(CK_HAS_AVX512VNNI)
    #define CK_QGEMM_DISPATCH(...) qgemm_avx512vnni(__VA_ARGS__)
#elif defined(CK_HAS_VNNI)
    #define CK_QGEMM_DISPATCH(...) qgemm_vnni(__VA_ARGS__)
#elif defined(CK_HAS_AVX2_FMA)
    #define CK_QGEMM_DISPATCH(...) qgemm_avx2(__VA_ARGS__)
#else
    #define CK_QGEMM_DISPATCH(...) qgemm_ref(__VA_ARGS__)
#endif

/*============================================================================
 * Capability Reporting
 *============================================================================*/

/**
 * @brief CPU capability information structure
 */
typedef struct {
    const char *name;
    int width;           /* Vector width in bits */
    int has_fma;         /* Fused multiply-add */
    int has_ai_accel;    /* AI-specific instructions (VNNI/AMX) */
    const char *best_kernel;  /* Recommended kernel name */
} ck_capability_t;

/**
 * @brief Get current platform capabilities
 */
static inline ck_capability_t ck_get_capabilities(void) {
    ck_capability_t cap = {
        .name = "unknown",
        .width = 32,
        .has_fma = 0,
        .has_ai_accel = 0,
        .best_kernel = "gemm_ref"
    };

#if defined(CK_HAS_AMX)
    cap.name = "AMX (Intel Sapphire Rapids+)";
    cap.width = 512;
    cap.has_fma = 1;
    cap.has_ai_accel = 1;
    cap.best_kernel = "gemm_amx";
#elif defined(CK_HAS_AVX512)
    cap.name = "AVX-512 (Intel Skylake-X+)";
    cap.width = 512;
    cap.has_fma = 1;
    cap.has_ai_accel = 1;
    cap.best_kernel = "gemm_avx512";
#elif defined(CK_HAS_AVX2_FMA)
    cap.name = "AVX2+FMA (Intel Haswell+)";
    cap.width = 256;
    cap.has_fma = 1;
    cap.has_ai_accel = 0;
    cap.best_kernel = "gemm_avx2";
#elif defined(CK_HAS_AVX)
    cap.name = "AVX (Intel Sandy Bridge+)";
    cap.width = 256;
    cap.has_fma = 0;
    cap.has_ai_accel = 0;
    cap.best_kernel = "gemm_avx";
#elif defined(CK_HAS_NEON)
    cap.name = "NEON (ARM)";
    cap.width = 128;
    cap.has_fma = 1;
    cap.has_ai_accel = 0;
    cap.best_kernel = "gemm_neon";
#elif defined(CK_HAS_ALTIVEC)
    cap.name = "AltiVec (PowerPC)";
    cap.width = 128;
    cap.has_fma = 1;
    cap.has_ai_accel = 0;
    cap.best_kernel = "gemm_altivec";
#endif

    return cap;
}

#endif /* CK_FEATURES_H */

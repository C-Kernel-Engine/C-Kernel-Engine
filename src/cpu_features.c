/**
 * CPU Feature Detection and Cache-Aware Parameter Tuning
 *
 * Detects CPU features, cache sizes, and core counts at runtime.
 * Computes optimal GEMM blocking parameters based on actual hardware.
 */

#include "cpu_features.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#ifdef _WIN32
#include <windows.h>
#include <intrin.h>
#else
#include <unistd.h>
#endif

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#define X86_CPU 1
#if defined(__GNUC__) || defined(__clang__)
#include <cpuid.h>
#endif
#endif

// Global instances
CPUInfo g_cpu_info = {0};
GEMMParams g_gemm_params = {0};
int g_cpu_initialized = 0;

// =============================================================================
// CPUID helpers for x86
// =============================================================================

#ifdef X86_CPU
static void cpuid(int leaf, int subleaf, uint32_t* eax, uint32_t* ebx, uint32_t* ecx, uint32_t* edx) {
#if defined(__GNUC__) || defined(__clang__)
    __cpuid_count(leaf, subleaf, *eax, *ebx, *ecx, *edx);
#elif defined(_MSC_VER)
    int regs[4];
    __cpuidex(regs, leaf, subleaf);
    *eax = regs[0]; *ebx = regs[1]; *ecx = regs[2]; *edx = regs[3];
#else
    *eax = *ebx = *ecx = *edx = 0;
#endif
}

static uint64_t xgetbv0(void) {
#if defined(__GNUC__) || defined(__clang__)
    uint32_t eax = 0, edx = 0;
    __asm__ __volatile__(".byte 0x0f, 0x01, 0xd0" : "=a"(eax), "=d"(edx) : "c"(0));
    return ((uint64_t)edx << 32) | eax;
#elif defined(_MSC_VER)
    return _xgetbv(0);
#else
    return 0;
#endif
}

static void detect_x86_features(CPUInfo* info) {
    uint32_t eax, ebx, ecx, edx;

    // Check max CPUID leaf
    cpuid(0, 0, &eax, &ebx, &ecx, &edx);
    uint32_t max_leaf = eax;

    if (max_leaf >= 1) {
        cpuid(1, 0, &eax, &ebx, &ecx, &edx);
        int has_osxsave = (ecx >> 27) & 1;
        uint64_t xcr0 = has_osxsave ? xgetbv0() : 0;
        int os_avx = ((xcr0 & 0x6) == 0x6);
        int os_avx512 = ((xcr0 & 0xE6) == 0xE6);
        int os_amx = ((xcr0 & 0x60000) == 0x60000);

        info->has_avx = ((ecx >> 28) & 1) && os_avx;
        info->has_fma = (ecx >> 12) & 1;

        if (max_leaf >= 7) {
            cpuid(7, 0, &eax, &ebx, &ecx, &edx);
            info->has_avx2 = ((ebx >> 5) & 1) && os_avx;
            info->has_avx512f = ((ebx >> 16) & 1) && os_avx512;
            info->has_avx512dq = ((ebx >> 17) & 1) && os_avx512;
            info->has_avx512bw = ((ebx >> 30) & 1) && os_avx512;
            info->has_avx512vl = ((ebx >> 31) & 1) && os_avx512;
            info->has_avx512_vnni = ((ecx >> 11) & 1) && os_avx512;
            info->has_amx_bf16 = ((edx >> 22) & 1) && os_amx;
            info->has_amx_tile = ((edx >> 24) & 1) && os_amx;
            info->has_amx_int8 = ((edx >> 25) & 1) && os_amx;
        }
    }

    if (max_leaf >= 7) {
        cpuid(7, 1, &eax, &ebx, &ecx, &edx);
        if (info->has_avx512f) {
            info->has_avx512_bf16 = (eax >> 5) & 1;
        }
    }
}

// Detect cache sizes using CPUID leaf 0x04 (Intel) or leaf 0x8000001D (AMD)
static void detect_x86_cache_sizes(CPUInfo* info) {
    uint32_t eax, ebx, ecx, edx;

    // Try Intel deterministic cache parameters (leaf 0x04)
    for (int index = 0; index < 16; index++) {
        cpuid(0x04, index, &eax, &ebx, &ecx, &edx);

        int cache_type = eax & 0x1F;
        if (cache_type == 0) break;  // No more caches

        int cache_level = (eax >> 5) & 0x7;
        int line_size = (ebx & 0xFFF) + 1;
        int partitions = ((ebx >> 12) & 0x3FF) + 1;
        int ways = ((ebx >> 22) & 0x3FF) + 1;
        int sets = ecx + 1;

        size_t cache_size = (size_t)line_size * partitions * ways * sets;

        if (cache_type == 1 || cache_type == 3) {  // Data or unified cache
            if (cache_level == 1) {
                info->l1d_size = cache_size;
                info->l1_line_size = line_size;
            } else if (cache_level == 2) {
                info->l2_size = cache_size;
            } else if (cache_level == 3) {
                info->l3_size = cache_size;
            }
        }
    }

    // If Intel method didn't work, try AMD method (leaf 0x8000001D)
    if (info->l1d_size == 0) {
        cpuid(0x80000000, 0, &eax, &ebx, &ecx, &edx);
        if (eax >= 0x8000001D) {
            for (int index = 0; index < 16; index++) {
                cpuid(0x8000001D, index, &eax, &ebx, &ecx, &edx);

                int cache_type = eax & 0x1F;
                if (cache_type == 0) break;

                int cache_level = (eax >> 5) & 0x7;
                int line_size = (ebx & 0xFFF) + 1;
                int partitions = ((ebx >> 12) & 0x3FF) + 1;
                int ways = ((ebx >> 22) & 0x3FF) + 1;
                int sets = ecx + 1;

                size_t cache_size = (size_t)line_size * partitions * ways * sets;

                if (cache_type == 1 || cache_type == 3) {
                    if (cache_level == 1) {
                        info->l1d_size = cache_size;
                        info->l1_line_size = line_size;
                    } else if (cache_level == 2) {
                        info->l2_size = cache_size;
                    } else if (cache_level == 3) {
                        info->l3_size = cache_size;
                    }
                }
            }
        }
    }
}
#endif

// =============================================================================
// Linux sysfs fallback for cache detection
// =============================================================================

#if defined(__linux__)
static size_t read_sysfs_cache_size(int cpu, int index) {
    char path[256];
    snprintf(path, sizeof(path),
             "/sys/devices/system/cpu/cpu%d/cache/index%d/size", cpu, index);

    FILE* f = fopen(path, "r");
    if (!f) return 0;

    char buf[32];
    if (!fgets(buf, sizeof(buf), f)) {
        fclose(f);
        return 0;
    }
    fclose(f);

    size_t size = 0;
    char unit = 'K';
    sscanf(buf, "%zu%c", &size, &unit);

    if (unit == 'K' || unit == 'k') size *= 1024;
    else if (unit == 'M' || unit == 'm') size *= 1024 * 1024;

    return size;
}

static int read_sysfs_cache_level(int cpu, int index) {
    char path[256];
    snprintf(path, sizeof(path),
             "/sys/devices/system/cpu/cpu%d/cache/index%d/level", cpu, index);

    FILE* f = fopen(path, "r");
    if (!f) return -1;

    int level = 0;
    if (fscanf(f, "%d", &level) != 1) level = -1;
    fclose(f);
    return level;
}

static void detect_linux_cache_sizes(CPUInfo* info) {
    // Read from CPU 0's cache info
    for (int index = 0; index < 10; index++) {
        size_t size = read_sysfs_cache_size(0, index);
        if (size == 0) break;

        int level = read_sysfs_cache_level(0, index);

        // Check if it's data or unified cache
        char path[256];
        snprintf(path, sizeof(path),
                 "/sys/devices/system/cpu/cpu0/cache/index%d/type", index);
        FILE* f = fopen(path, "r");
        if (f) {
            char type[32] = {0};
            if (fscanf(f, "%31s", type) != 1) type[0] = '\0';
            fclose(f);

            if (strcmp(type, "Data") == 0 || strcmp(type, "Unified") == 0) {
                if (level == 1) info->l1d_size = size;
                else if (level == 2) info->l2_size = size;
                else if (level == 3) info->l3_size = size;
            }
        }
    }
}

static int detect_linux_physical_cores(void) {
    // Count unique physical cores by reading core_id
    int max_core_id = -1;
    int num_processors = 0;

    FILE* f = fopen("/proc/cpuinfo", "r");
    if (!f) return 1;

    char line[256];
    while (fgets(line, sizeof(line), f)) {
        if (strncmp(line, "processor", 9) == 0) {
            num_processors++;
        }
        if (strncmp(line, "cpu cores", 9) == 0) {
            int cores = 0;
            sscanf(line, "cpu cores : %d", &cores);
            if (cores > 0) {
                fclose(f);
                return cores;
            }
        }
        if (strncmp(line, "core id", 7) == 0) {
            int core_id = 0;
            sscanf(line, "core id : %d", &core_id);
            if (core_id > max_core_id) max_core_id = core_id;
        }
    }
    fclose(f);

    // Fallback: use processor count (may include hyperthreads)
    if (max_core_id >= 0) {
        return max_core_id + 1;
    }
    return num_processors > 0 ? num_processors : 1;
}
#endif

// =============================================================================
// Physical core detection
// =============================================================================

static int detect_physical_cores(void) {
#if defined(__linux__)
    return detect_linux_physical_cores();
#elif defined(_WIN32)
    SYSTEM_INFO sysinfo;
    GetSystemInfo(&sysinfo);
    // Windows: this gives logical processors, divide by 2 for HT estimate
    return sysinfo.dwNumberOfProcessors / 2;
#elif defined(__APPLE__)
    int cores = 1;
    size_t len = sizeof(cores);
    sysctlbyname("hw.physicalcpu", &cores, &len, NULL, 0);
    return cores;
#else
    return 1;
#endif
}

// =============================================================================
// Compute optimal GEMM blocking parameters
// =============================================================================

static void compute_gemm_params(const CPUInfo* cpu, GEMMParams* params) {
    // Microkernel sizes based on SIMD width
    // MUST match compile-time MR_FIXED/NR_FIXED in gemm_microkernel.c
    if (cpu->has_avx512f) {
        params->MR = 6;   // 6 rows
        params->NR = 32;  // 32 cols (2 x ZMM registers)
    } else if (cpu->has_fma) {
        // AVX2+FMA: can use 6x16 with FMA hiding register spilling
        params->MR = 6;   // 6 rows
        params->NR = 16;  // 16 cols (2 x YMM registers)
    } else if (cpu->has_avx || cpu->has_avx2) {
        // AVX without FMA: use 4x16 to avoid register spilling
        params->MR = 4;   // 4 rows (reduced to fit in 16 YMM registers)
        params->NR = 16;  // 16 cols (2 x YMM registers)
    } else {
        params->MR = 4;
        params->NR = 4;
    }

    // Get cache sizes (use defaults if detection failed)
    size_t l1 = cpu->l1d_size > 0 ? cpu->l1d_size : 32 * 1024;   // Default 32KB
    size_t l2 = cpu->l2_size > 0 ? cpu->l2_size : 256 * 1024;    // Default 256KB
    size_t l3 = cpu->l3_size > 0 ? cpu->l3_size : 8 * 1024 * 1024; // Default 8MB

    // BLIS-style blocking parameter computation
    // Reference: "Anatomy of High-Performance Matrix Multiplication" (Goto & Van de Geijn)
    //
    // KC: Controls L1 usage
    //   - A micropanel: MR x KC
    //   - B micropanel: KC x NR (streamed from L2)
    //   - Want MR * KC * sizeof(float) to fit in ~half of L1
    //
    // MC: Controls L2 usage
    //   - A block: MC x KC should fit in L2
    //
    // NC: Controls L3 usage / main memory streaming
    //   - B panel: KC x NC

    // KC: Controls L1 usage
    // For optimal performance, both A micropanel and B row should fit in L1:
    //   - A micropanel: MR * KC floats
    //   - B row for streaming: NR floats per iteration (small)
    // Use ~25% of L1 for A micropanel to leave room for B and working set
    size_t l1_for_a = (l1 * 25) / 100;
    params->KC = (int)(l1_for_a / (params->MR * sizeof(float)));

    // Round KC to multiple of 8 for alignment
    params->KC = (params->KC / 8) * 8;
    if (params->KC < 64) params->KC = 64;
    if (params->KC > 512) params->KC = 512;  // Cap at 512 for better cache fit

    // MC: A block = MC * KC * 4 bytes should fit in ~80% of L2
    size_t l2_for_a = (l2 * 80) / 100;
    params->MC = (int)(l2_for_a / (params->KC * sizeof(float)));

    // Round MC to multiple of MR
    params->MC = (params->MC / params->MR) * params->MR;
    if (params->MC < params->MR * 4) params->MC = params->MR * 4;
    if (params->MC > 512) params->MC = 512;

    // NC: B panel = KC * NC * 4 bytes
    // For L3, we want good streaming, use ~50% of L3 / num_cores
    size_t l3_per_core = l3 / (cpu->num_cores > 0 ? cpu->num_cores : 1);
    size_t l3_for_b = (l3_per_core * 50) / 100;
    params->NC = (int)(l3_for_b / (params->KC * sizeof(float)));

    // Round NC to multiple of NR
    params->NC = (params->NC / params->NR) * params->NR;
    if (params->NC < params->NR * 8) params->NC = params->NR * 8;
    if (params->NC > 8192) params->NC = 8192;
}

// =============================================================================
// Public API
// =============================================================================

void cpu_features_init(void) {
    if (g_cpu_initialized) return;

    memset(&g_cpu_info, 0, sizeof(g_cpu_info));
    memset(&g_gemm_params, 0, sizeof(g_gemm_params));

    // Detect SIMD features
#ifdef X86_CPU
    detect_x86_features(&g_cpu_info);
    detect_x86_cache_sizes(&g_cpu_info);
#endif

    // Linux sysfs fallback for cache sizes
#if defined(__linux__)
    if (g_cpu_info.l1d_size == 0) {
        detect_linux_cache_sizes(&g_cpu_info);
    }
#endif

    // Detect physical cores
    g_cpu_info.num_cores = detect_physical_cores();

    // Compute GEMM parameters based on detected hardware
    compute_gemm_params(&g_cpu_info, &g_gemm_params);

    g_cpu_initialized = 1;
}

const GEMMParams* get_gemm_params(void) {
    if (!g_cpu_initialized) cpu_features_init();
    return &g_gemm_params;
}

const CPUInfo* get_cpu_info(void) {
    if (!g_cpu_initialized) cpu_features_init();
    return &g_cpu_info;
}

void print_cpu_info(void) {
    if (!g_cpu_initialized) cpu_features_init();

    printf("=== CPU Info ===\n");
    printf("Physical cores: %d\n", g_cpu_info.num_cores);
    printf("L1 Data Cache:  %zu KB\n", g_cpu_info.l1d_size / 1024);
    printf("L2 Cache:       %zu KB\n", g_cpu_info.l2_size / 1024);
    printf("L3 Cache:       %zu MB\n", g_cpu_info.l3_size / (1024 * 1024));
    printf("Cache line:     %zu bytes\n", g_cpu_info.l1_line_size);
    printf("AVX:            %s\n", g_cpu_info.has_avx ? "yes" : "no");
    printf("AVX2:           %s\n", g_cpu_info.has_avx2 ? "yes" : "no");
    printf("AVX-512F:       %s\n", g_cpu_info.has_avx512f ? "yes" : "no");
    printf("AVX-512BW/DQ/VL:%s/%s/%s\n",
           g_cpu_info.has_avx512bw ? "yes" : "no",
           g_cpu_info.has_avx512dq ? "yes" : "no",
           g_cpu_info.has_avx512vl ? "yes" : "no");
    printf("AVX-512 VNNI:   %s\n", g_cpu_info.has_avx512_vnni ? "yes" : "no");
    printf("AVX-512 BF16:   %s\n", g_cpu_info.has_avx512_bf16 ? "yes" : "no");
    printf("AMX tile/int8/bf16: %s/%s/%s\n",
           g_cpu_info.has_amx_tile ? "yes" : "no",
           g_cpu_info.has_amx_int8 ? "yes" : "no",
           g_cpu_info.has_amx_bf16 ? "yes" : "no");
    printf("FMA:            %s\n", g_cpu_info.has_fma ? "yes" : "no");
    printf("\n=== GEMM Blocking Parameters ===\n");
    printf("MR (microkernel rows): %d\n", g_gemm_params.MR);
    printf("NR (microkernel cols): %d\n", g_gemm_params.NR);
    printf("MC (M block):          %d\n", g_gemm_params.MC);
    printf("NC (N block):          %d\n", g_gemm_params.NC);
    printf("KC (K block):          %d\n", g_gemm_params.KC);
    printf("\n");
}

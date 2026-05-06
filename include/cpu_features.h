/**
 * CPU Feature Detection and Cache-Aware Parameter Tuning
 *
 * This module detects CPU cache sizes at runtime and computes optimal
 * GEMM blocking parameters based on the BLIS/oneDNN methodology.
 *
 * Key formulas:
 * - KC: MR * KC * sizeof(float) fits in L1 (with room for B streaming)
 * - MC: MC * KC * sizeof(float) fits in L2
 * - NC: KC * NR fits in L1 for B panel, NC scales with L3
 */

#ifndef CPU_FEATURES_H
#define CPU_FEATURES_H

#include <stdint.h>
#include <stddef.h>

// CPU cache information
typedef struct {
    size_t l1d_size;      // L1 data cache size in bytes
    size_t l2_size;       // L2 cache size in bytes
    size_t l3_size;       // L3 cache size in bytes
    size_t l1_line_size;  // Cache line size
    int num_cores;        // Number of physical cores
    int has_avx;
    int has_avx2;
    int has_avx512f;
    int has_avx512bw;
    int has_avx512dq;
    int has_avx512vl;
    int has_avx512_vnni;
    int has_avx512_bf16;
    int has_amx_tile;
    int has_amx_int8;
    int has_amx_bf16;
    int has_fma;
} CPUInfo;

// GEMM blocking parameters (computed based on cache sizes)
typedef struct {
    int MR;   // Microkernel rows
    int NR;   // Microkernel cols
    int MC;   // M block size
    int NC;   // N block size
    int KC;   // K block size
} GEMMParams;

// Global CPU info and GEMM params (initialized once at startup)
extern CPUInfo g_cpu_info;
extern GEMMParams g_gemm_params;
extern int g_cpu_initialized;

// Initialize CPU detection (call once at startup)
void cpu_features_init(void);

// Get current GEMM parameters
const GEMMParams* get_gemm_params(void);

// Get CPU info
const CPUInfo* get_cpu_info(void);

// Debug: print CPU info
void print_cpu_info(void);

#endif // CPU_FEATURES_H

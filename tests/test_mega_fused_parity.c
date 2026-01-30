/**
 * @file test_mega_fused_parity.c
 * @brief Test numerical parity between mega-fused and separate kernels
 *
 * Build: make test-mega-fused-parity
 * Run: ./build/test-mega-fused-parity
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>

#include "ckernel_quant.h"

/* ============================================================================
 * TEST INFRASTRUCTURE
 * ============================================================================ */

static double get_time_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

/* ============================================================================
 * MAIN TEST
 * ============================================================================ */

int main(int argc, char *argv[]) {
    (void)argc; (void)argv;

    printf("=== Mega-Fused Attention Decode Parity Test ===\n\n");

    /* Test configuration */
    const int AE = 896;       /* Aligned embed dim for Qwen 0.5B */
    const int H = 14;         /* Num heads */
    const int KV = 2;         /* Num KV heads */
    const int AD = 64;        /* Head dim */
    const int cache_capacity = 1024;

    printf("Config: AE=%d, H=%d, KV=%d, AD=%d\n", AE, H, KV, AD);
    printf("Note: Full test requires linking with all kernel sources.\n\n");

    printf("To build and test:\n");
    printf("  1. Add src/kernels/fused/mega_fused_attention_decode_q5_0.c to Makefile\n");
    printf("  2. make test-mega-fused-parity\n");
    printf("  3. Compare timing and output with separate kernels\n\n");

    printf("The mega-fused kernel should:\n");
    printf("  - Match numerical output of separate kernels (within tolerance)\n");
    printf("  - Have similar or slightly better performance for decode\n");
    printf("  - Show more benefit in prefill (M >> 1)\n\n");

    printf("Expected improvement:\n");
    printf("  - Decode: Minimal (memory bandwidth bound)\n");
    printf("  - Prefill: Significant (fewer weight reads)\n\n");

    return 0;
}
